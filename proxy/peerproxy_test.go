package proxy

import (
	"context"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mostlygeek/llama-swap/proxy/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewPeerProxy_EmptyPeers(t *testing.T) {
	peers := config.PeerDictionaryConfig{}
	pm, err := NewPeerProxy(peers, testLogger)
	require.NoError(t, err)
	assert.NotNil(t, pm)
	assert.Empty(t, pm.proxyMap)
}

func TestNewPeerProxy_SinglePeer(t *testing.T) {
	proxyURL, _ := url.Parse("http://peer1.example.com:8080")
	peers := config.PeerDictionaryConfig{
		"peer1": config.PeerConfig{
			Proxy:    "http://peer1.example.com:8080",
			ProxyURL: proxyURL,
			ApiKey:   "test-key",
			Models:   []string{"model-a", "model-b"},
		},
	}

	pm, err := NewPeerProxy(peers, testLogger)
	require.NoError(t, err)
	assert.Len(t, pm.proxyMap, 2)
	assert.True(t, pm.HasPeerModel("model-a"))
	assert.True(t, pm.HasPeerModel("model-b"))
	assert.False(t, pm.HasPeerModel("model-c"))
}

func TestNewPeerProxy_MultiplePeers(t *testing.T) {
	proxyURL1, _ := url.Parse("http://peer1.example.com:8080")
	proxyURL2, _ := url.Parse("http://peer2.example.com:8080")
	peers := config.PeerDictionaryConfig{
		"peer1": config.PeerConfig{
			Proxy:    "http://peer1.example.com:8080",
			ProxyURL: proxyURL1,
			Models:   []string{"model-a", "model-b"},
		},
		"peer2": config.PeerConfig{
			Proxy:    "http://peer2.example.com:8080",
			ProxyURL: proxyURL2,
			Models:   []string{"model-c", "model-d"},
		},
	}

	pm, err := NewPeerProxy(peers, testLogger)
	require.NoError(t, err)
	assert.Len(t, pm.proxyMap, 4)
	assert.True(t, pm.HasPeerModel("model-a"))
	assert.True(t, pm.HasPeerModel("model-b"))
	assert.True(t, pm.HasPeerModel("model-c"))
	assert.True(t, pm.HasPeerModel("model-d"))
}

func TestNewPeerProxy_DuplicateModelWarning(t *testing.T) {
	// When the same model is in multiple peers, only the first (lexicographically by peer ID)
	// should be mapped, and a warning should be logged
	proxyURL1, _ := url.Parse("http://peer1.example.com:8080")
	proxyURL2, _ := url.Parse("http://peer2.example.com:8080")
	peers := config.PeerDictionaryConfig{
		"alpha-peer": config.PeerConfig{
			Proxy:    "http://peer1.example.com:8080",
			ProxyURL: proxyURL1,
			Models:   []string{"duplicate-model"},
		},
		"beta-peer": config.PeerConfig{
			Proxy:    "http://peer2.example.com:8080",
			ProxyURL: proxyURL2,
			Models:   []string{"duplicate-model"},
		},
	}

	pm, err := NewPeerProxy(peers, testLogger)
	require.NoError(t, err)
	// Should only have one entry for the duplicate model
	assert.Len(t, pm.proxyMap, 1)
	assert.True(t, pm.HasPeerModel("duplicate-model"))
}

func TestHasPeerModel(t *testing.T) {
	proxyURL, _ := url.Parse("http://peer1.example.com:8080")
	peers := config.PeerDictionaryConfig{
		"peer1": config.PeerConfig{
			Proxy:    "http://peer1.example.com:8080",
			ProxyURL: proxyURL,
			Models:   []string{"existing-model"},
		},
	}

	pm, err := NewPeerProxy(peers, testLogger)
	require.NoError(t, err)

	assert.True(t, pm.HasPeerModel("existing-model"))
	assert.False(t, pm.HasPeerModel("non-existing-model"))
}

func TestProxyRequest_ModelNotFound(t *testing.T) {
	peers := config.PeerDictionaryConfig{}
	pm, err := NewPeerProxy(peers, testLogger)
	require.NoError(t, err)

	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	w := httptest.NewRecorder()

	err = pm.ProxyRequest("non-existing-model", w, req)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no peer proxy found for model non-existing-model")
}

func TestProxyRequest_Success(t *testing.T) {
	// Create a test server to act as the peer
	testServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("response from peer"))
	}))
	defer testServer.Close()

	proxyURL, _ := url.Parse(testServer.URL)
	peers := config.PeerDictionaryConfig{
		"peer1": config.PeerConfig{
			Proxy:    testServer.URL,
			ProxyURL: proxyURL,
			Models:   []string{"test-model"},
		},
	}

	pm, err := NewPeerProxy(peers, testLogger)
	require.NoError(t, err)

	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	w := httptest.NewRecorder()

	err = pm.ProxyRequest("test-model", w, req)
	assert.NoError(t, err)
	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, "response from peer", w.Body.String())
}

func TestProxyRequest_ApiKeyInjection(t *testing.T) {
	// Create a test server that checks for the Authorization header
	var receivedAuthHeader string
	testServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedAuthHeader = r.Header.Get("Authorization")
		w.WriteHeader(http.StatusOK)
	}))
	defer testServer.Close()

	proxyURL, _ := url.Parse(testServer.URL)
	peers := config.PeerDictionaryConfig{
		"peer1": config.PeerConfig{
			Proxy:    testServer.URL,
			ProxyURL: proxyURL,
			ApiKey:   "secret-api-key",
			Models:   []string{"test-model"},
		},
	}

	pm, err := NewPeerProxy(peers, testLogger)
	require.NoError(t, err)

	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	w := httptest.NewRecorder()

	err = pm.ProxyRequest("test-model", w, req)
	assert.NoError(t, err)
	assert.Equal(t, "Bearer secret-api-key", receivedAuthHeader)
}

func TestProxyRequest_NoApiKey(t *testing.T) {
	// Create a test server that checks for the Authorization header
	var receivedAuthHeader string
	testServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedAuthHeader = r.Header.Get("Authorization")
		w.WriteHeader(http.StatusOK)
	}))
	defer testServer.Close()

	proxyURL, _ := url.Parse(testServer.URL)
	peers := config.PeerDictionaryConfig{
		"peer1": config.PeerConfig{
			Proxy:    testServer.URL,
			ProxyURL: proxyURL,
			ApiKey:   "", // No API key
			Models:   []string{"test-model"},
		},
	}

	pm, err := NewPeerProxy(peers, testLogger)
	require.NoError(t, err)

	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	w := httptest.NewRecorder()

	err = pm.ProxyRequest("test-model", w, req)
	assert.NoError(t, err)
	assert.Empty(t, receivedAuthHeader)
}

func TestProxyRequest_HostHeaderSet(t *testing.T) {
	// Create a test server that checks the Host header
	var receivedHost string
	testServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedHost = r.Host
		w.WriteHeader(http.StatusOK)
	}))
	defer testServer.Close()

	proxyURL, _ := url.Parse(testServer.URL)
	peers := config.PeerDictionaryConfig{
		"peer1": config.PeerConfig{
			Proxy:    testServer.URL,
			ProxyURL: proxyURL,
			Models:   []string{"test-model"},
		},
	}

	pm, err := NewPeerProxy(peers, testLogger)
	require.NoError(t, err)

	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	w := httptest.NewRecorder()

	err = pm.ProxyRequest("test-model", w, req)
	assert.NoError(t, err)
	// The Host header should be set to the target URL's host
	assert.True(t, strings.HasPrefix(receivedHost, "127.0.0.1:"))
}

func TestProxyRequest_SSEHeaderModification(t *testing.T) {
	// Create a test server that returns SSE content type
	testServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
	}))
	defer testServer.Close()

	proxyURL, _ := url.Parse(testServer.URL)
	peers := config.PeerDictionaryConfig{
		"peer1": config.PeerConfig{
			Proxy:    testServer.URL,
			ProxyURL: proxyURL,
			Models:   []string{"test-model"},
		},
	}

	pm, err := NewPeerProxy(peers, testLogger)
	require.NoError(t, err)

	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	w := httptest.NewRecorder()

	err = pm.ProxyRequest("test-model", w, req)
	assert.NoError(t, err)
	// The X-Accel-Buffering header should be set to "no" for SSE
	assert.Equal(t, "no", w.Header().Get("X-Accel-Buffering"))
}

func TestNewPeerProxy_CustomTimeouts(t *testing.T) {
	proxyURL, _ := url.Parse("http://localhost:8080")

	peers := config.PeerDictionaryConfig{
		"test-peer": config.PeerConfig{
			Proxy:    "http://localhost:8080",
			ProxyURL: proxyURL,
			Models:   []string{"model1"},
			Timeouts: config.TimeoutsConfig{
				Connect:        45,
				ResponseHeader: 300,
				TLSHandshake:   15,
				ExpectContinue: 2,
				IdleConn:       120,
			},
		},
	}

	peerProxy, err := NewPeerProxy(peers, testLogger)

	assert.NoError(t, err)
	assert.NotNil(t, peerProxy)
	assert.True(t, peerProxy.HasPeerModel("model1"))

	// Verify the timeout values are actually applied to the transport
	member, found := peerProxy.proxyMap["model1"]
	require.True(t, found, "model1 should exist in proxyMap")
	assert.NotNil(t, member.reverseProxy)
	assert.NotNil(t, member.reverseProxy.Transport)

	transport, ok := member.reverseProxy.Transport.(*http.Transport)
	require.True(t, ok, "Transport should be *http.Transport")

	// Verify all timeout values are correctly applied
	assert.Equal(t, 300*time.Second, transport.ResponseHeaderTimeout)
	assert.Equal(t, 15*time.Second, transport.TLSHandshakeTimeout)
	assert.Equal(t, 2*time.Second, transport.ExpectContinueTimeout)
	assert.Equal(t, 120*time.Second, transport.IdleConnTimeout)
	// ForceAttemptHTTP2 should be enabled
	assert.True(t, transport.ForceAttemptHTTP2)
}

func TestBuildMagicPacket(t *testing.T) {
	packet, err := buildMagicPacket(net.HardwareAddr{0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff})
	require.NoError(t, err)
	require.Len(t, packet, 102)

	for i := 0; i < 6; i++ {
		assert.Equal(t, byte(0xff), packet[i])
	}
	for offset := 6; offset < len(packet); offset += 6 {
		assert.Equal(t, []byte{0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff}, packet[offset:offset+6])
	}
}

func TestSendMagicPacket(t *testing.T) {
	listener, err := net.ListenUDP("udp4", &net.UDPAddr{IP: net.IPv4(127, 0, 0, 1), Port: 0})
	require.NoError(t, err)
	defer listener.Close()

	packetCh := make(chan []byte, 1)
	go func() {
		buf := make([]byte, 256)
		_ = listener.SetReadDeadline(time.Now().Add(2 * time.Second))
		n, _, err := listener.ReadFromUDP(buf)
		if err == nil {
			packetCh <- append([]byte(nil), buf[:n]...)
		}
	}()

	err = sendMagicPacketToPort(config.WakeOnLanConfig{
		MAC:       "aa:bb:cc:dd:ee:ff",
		Broadcast: "127.0.0.1",
	}, listener.LocalAddr().(*net.UDPAddr).Port)
	require.NoError(t, err)

	select {
	case packet := <-packetCh:
		require.Len(t, packet, 102)
		for i := 0; i < 6; i++ {
			assert.Equal(t, byte(0xff), packet[i])
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for wake packet")
	}
}

func TestProxyRequest_WakeOnLan_WaitsForReachability(t *testing.T) {
	peerServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"response":"ready"}`))
	}))
	defer peerServer.Close()

	proxyURL, _ := url.Parse(peerServer.URL)
	peers := config.PeerDictionaryConfig{
		"peer1": config.PeerConfig{
			Proxy:    peerServer.URL,
			ProxyURL: proxyURL,
			Models:   []string{"sleeping-model"},
			WakeOnLan: &config.WakeOnLanConfig{
				MAC: "aa:bb:cc:dd:ee:ff",
			},
		},
	}

	pm, err := NewPeerProxy(peers, testLogger)
	require.NoError(t, err)
	pm.probeReachability = func(ctx context.Context, target *url.URL) error {
		return errors.New("unreachable")
	}

	wakeTriggered := atomic.Bool{}
	pm.sendWakePacket = func(wol config.WakeOnLanConfig) error {
		wakeTriggered.Store(true)
		pm.probeReachability = func(ctx context.Context, target *url.URL) error {
			if wakeTriggered.Load() {
				return nil
			}
			return errors.New("unreachable")
		}
		return nil
	}

	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	w := httptest.NewRecorder()

	err = pm.ProxyRequest("sleeping-model", w, req)
	require.NoError(t, err)
	assert.True(t, wakeTriggered.Load())
	assert.Equal(t, http.StatusOK, w.Code)
	assert.Contains(t, w.Body.String(), "ready")
}

func TestProxyRequest_WakeOnLan_SharedWakeAttempt(t *testing.T) {
	peerServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"response":"ready"}`))
	}))
	defer peerServer.Close()

	proxyURL, _ := url.Parse(peerServer.URL)
	peers := config.PeerDictionaryConfig{
		"peer1": config.PeerConfig{
			Proxy:    peerServer.URL,
			ProxyURL: proxyURL,
			Models:   []string{"sleeping-model"},
			WakeOnLan: &config.WakeOnLanConfig{
				MAC: "aa:bb:cc:dd:ee:ff",
			},
		},
	}

	pm, err := NewPeerProxy(peers, testLogger)
	require.NoError(t, err)

	var wakeCount atomic.Int32
	ready := make(chan struct{})
	readyOnce := sync.Once{}

	pm.probeReachability = func(ctx context.Context, target *url.URL) error {
		select {
		case <-ready:
			return nil
		default:
			return errors.New("unreachable")
		}
	}

	pm.sendWakePacket = func(wol config.WakeOnLanConfig) error {
		if wakeCount.Add(1) == 1 {
			readyOnce.Do(func() { close(ready) })
		}
		return nil
	}

	var wg sync.WaitGroup
	errs := make(chan error, 2)
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
			w := httptest.NewRecorder()
			errs <- pm.ProxyRequest("sleeping-model", w, req)
			require.Equal(t, http.StatusOK, w.Code)
		}()
	}

	wg.Wait()
	close(errs)
	for err := range errs {
		require.NoError(t, err)
	}

	assert.Equal(t, int32(1), wakeCount.Load())
}

func TestProxyRequest_WakeOnLan_Timeout(t *testing.T) {
	proxyURL, _ := url.Parse("http://127.0.0.1:65534")
	peers := config.PeerDictionaryConfig{
		"peer1": config.PeerConfig{
			Proxy:    "http://127.0.0.1:65534",
			ProxyURL: proxyURL,
			Models:   []string{"sleeping-model"},
			WakeOnLan: &config.WakeOnLanConfig{
				MAC: "aa:bb:cc:dd:ee:ff",
			},
		},
	}

	pm, err := NewPeerProxy(peers, testLogger)
	require.NoError(t, err)
	pm.wakeTimeout = 25 * time.Millisecond
	pm.probeReachability = func(ctx context.Context, target *url.URL) error {
		return errors.New("unreachable")
	}
	pm.sendWakePacket = func(wol config.WakeOnLanConfig) error { return nil }

	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	w := httptest.NewRecorder()

	err = pm.ProxyRequest("sleeping-model", w, req)
	require.NoError(t, err)
	assert.Equal(t, http.StatusGatewayTimeout, w.Code)
	assert.Contains(t, w.Body.String(), "peer wake timed out")
}
