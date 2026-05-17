package proxy

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/mostlygeek/llama-swap/internal/logmon"
	"github.com/mostlygeek/llama-swap/proxy/config"
)

var ErrPeerWakeTimeout = errors.New("peer wake timed out")

const (
	peerWakeTimeout   = 30 * time.Second
	peerProbeInterval = 1 * time.Second
	peerProbeTimeout  = 2 * time.Second
)

type peerProxyMember struct {
	peerID       string
	reverseProxy *httputil.ReverseProxy
	apiKey       string
}

type wakeState struct {
	done chan struct{}
	err  error
}

type PeerProxy struct {
	peers    config.PeerDictionaryConfig
	proxyMap map[string]*peerProxyMember

	wakeMu    sync.Mutex
	wakeState map[string]*wakeState

	wakeTimeout time.Duration

	probeReachability func(context.Context, *url.URL) error
	sendWakePacket    func(config.WakeOnLanConfig) error
}

func NewPeerProxy(peers config.PeerDictionaryConfig, proxyLogger *logmon.Monitor) (*PeerProxy, error) {
	proxyMap := make(map[string]*peerProxyMember)

	// Sort peer IDs for consistent iteration order
	peerIDs := make([]string, 0, len(peers))
	for peerID := range peers {
		peerIDs = append(peerIDs, peerID)
	}
	sort.Strings(peerIDs)

	pp := &PeerProxy{
		peers:             peers,
		proxyMap:          proxyMap,
		wakeState:         make(map[string]*wakeState),
		wakeTimeout:       peerWakeTimeout,
		probeReachability: probeTCPReachability,
		sendWakePacket:    sendMagicPacket,
	}

	for _, peerID := range peerIDs {
		peer := peers[peerID]

		// Create a transport with per-peer timeout configuration
		peerTransport := &http.Transport{
			Proxy: http.ProxyFromEnvironment,
			DialContext: (&net.Dialer{
				Timeout:   time.Duration(peer.Timeouts.Connect) * time.Second,
				KeepAlive: time.Duration(peer.Timeouts.KeepAlive) * time.Second,
			}).DialContext,
			TLSHandshakeTimeout:   time.Duration(peer.Timeouts.TLSHandshake) * time.Second,
			ResponseHeaderTimeout: time.Duration(peer.Timeouts.ResponseHeader) * time.Second,
			ExpectContinueTimeout: time.Duration(peer.Timeouts.ExpectContinue) * time.Second,
			ForceAttemptHTTP2:     true,
			MaxIdleConns:          100,
			MaxIdleConnsPerHost:   10,
			IdleConnTimeout:       time.Duration(peer.Timeouts.IdleConn) * time.Second,
		}

		// Create reverse proxy for this peer.
		reverseProxy := &httputil.ReverseProxy{
			Transport: peerTransport,
		}

		// Rewrite the outbound request to the peer and keep the Host header aligned
		// with the target URL for remote proxying.
		reverseProxy.Rewrite = func(pr *httputil.ProxyRequest) {
			pr.SetURL(peer.ProxyURL)
			pr.Out.Host = pr.Out.URL.Host
		}

		reverseProxy.ModifyResponse = func(resp *http.Response) error {
			if strings.Contains(strings.ToLower(resp.Header.Get("Content-Type")), "text/event-stream") {
				resp.Header.Set("X-Accel-Buffering", "no")
			}
			return nil
		}

		reverseProxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
			proxyLogger.Warnf("peer %s: proxy error: %v", peerID, err)
			errMsg := fmt.Sprintf("peer proxy error: %v", err)
			if runtime.GOOS == "darwin" && strings.Contains(err.Error(), "connect: no route to host") {
				errMsg += " (hint: on macOS, check System Settings > Privacy & Security > Local Network permissions)"
			}
			http.Error(w, errMsg, http.StatusBadGateway)
		}

		ppMember := &peerProxyMember{
			peerID:       peerID,
			reverseProxy: reverseProxy,
			apiKey:       peer.ApiKey,
		}

		// Map each model to this peer's proxy
		for _, modelID := range peer.Models {
			if _, found := proxyMap[modelID]; found {
				proxyLogger.Warnf("peer %s: model %s already mapped to another peer, skipping", peerID, modelID)
				continue
			}
			proxyMap[modelID] = ppMember
		}
	}

	return pp, nil
}

func (p *PeerProxy) HasPeerModel(modelID string) bool {
	_, found := p.proxyMap[modelID]
	return found
}

// GetPeerFilters returns the filters for a peer model, or empty filters if not found
func (p *PeerProxy) GetPeerFilters(modelID string) config.Filters {
	pp, found := p.proxyMap[modelID]
	if !found {
		return config.Filters{}
	}
	// Get the peer config using the peerID
	peer, found := p.peers[pp.peerID]
	if !found {
		return config.Filters{}
	}
	return peer.Filters
}

func (p *PeerProxy) ListPeers() config.PeerDictionaryConfig {
	return p.peers
}

func (p *PeerProxy) ProxyRequest(modelID string, writer http.ResponseWriter, request *http.Request) error {
	pp, found := p.proxyMap[modelID]
	if !found {
		return fmt.Errorf("no peer proxy found for model %s", modelID)
	}

	peer, found := p.peers[pp.peerID]
	if !found {
		return fmt.Errorf("peer configuration not found for model %s", modelID)
	}

	if err := p.ensurePeerAwake(request.Context(), pp.peerID, peer); err != nil {
		if errors.Is(err, ErrPeerWakeTimeout) {
			http.Error(writer, err.Error(), http.StatusGatewayTimeout)
			return nil
		}
		return err
	}

	// Inject API key if configured for this peer
	if pp.apiKey != "" {
		request.Header.Set("Authorization", "Bearer "+pp.apiKey)
		request.Header.Set("x-api-key", pp.apiKey)
	}

	pp.reverseProxy.ServeHTTP(writer, request)
	return nil
}

func (p *PeerProxy) ensurePeerAwake(ctx context.Context, peerID string, peer config.PeerConfig) error {
	if peer.WakeOnLan == nil {
		return nil
	}

	if err := p.probeReachabilityWithTimeout(ctx, peer.ProxyURL); err == nil {
		return nil
	}

	state := p.getOrStartWake(peerID, peer)
	select {
	case <-state.done:
		return state.err
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (p *PeerProxy) getOrStartWake(peerID string, peer config.PeerConfig) *wakeState {
	p.wakeMu.Lock()
	defer p.wakeMu.Unlock()

	if existing, ok := p.wakeState[peerID]; ok {
		return existing
	}

	state := &wakeState{done: make(chan struct{})}
	p.wakeState[peerID] = state

	go func() {
		defer close(state.done)
		defer p.clearWakeState(peerID)
		state.err = p.performWake(peer)
	}()

	return state
}

func (p *PeerProxy) clearWakeState(peerID string) {
	p.wakeMu.Lock()
	delete(p.wakeState, peerID)
	p.wakeMu.Unlock()
}

func (p *PeerProxy) performWake(peer config.PeerConfig) error {
	wakeTimeout := p.wakeTimeout
	if wakeTimeout <= 0 {
		wakeTimeout = peerWakeTimeout
	}

	ctx, cancel := context.WithTimeout(context.Background(), wakeTimeout)
	defer cancel()

	if err := p.sendWakePacket(*peer.WakeOnLan); err != nil {
		return err
	}

	ticker := time.NewTicker(peerProbeInterval)
	defer ticker.Stop()

	if err := p.probeReachabilityWithTimeout(ctx, peer.ProxyURL); err == nil {
		return nil
	}

	for {
		select {
		case <-ctx.Done():
			return ErrPeerWakeTimeout
		case <-ticker.C:
			if err := p.probeReachabilityWithTimeout(ctx, peer.ProxyURL); err == nil {
				return nil
			}
		}
	}
}

func (p *PeerProxy) probeReachabilityWithTimeout(ctx context.Context, target *url.URL) error {
	probeCtx, cancel := context.WithTimeout(ctx, peerProbeTimeout)
	defer cancel()
	return p.probeReachability(probeCtx, target)
}
