package proxy

import (
	"bytes"
	"context"
	"fmt"
	"net"
	"net/url"
	"strings"
	"syscall"

	"github.com/mostlygeek/llama-swap/proxy/config"
)

const (
	wolDefaultPort          = 9
	wolDefaultBroadcastAddr = "255.255.255.255"
)

func buildMagicPacket(mac net.HardwareAddr) ([]byte, error) {
	if len(mac) != 6 {
		return nil, fmt.Errorf("wake-on-lan requires a 6-byte MAC address, got %d bytes", len(mac))
	}

	packet := make([]byte, 6+16*len(mac))
	copy(packet[:6], bytes.Repeat([]byte{0xFF}, 6))
	for offset := 6; offset < len(packet); offset += len(mac) {
		copy(packet[offset:], mac)
	}

	return packet, nil
}

func sendMagicPacket(wol config.WakeOnLanConfig) error {
	return sendMagicPacketToPort(wol, wolDefaultPort)
}

func sendMagicPacketToPort(wol config.WakeOnLanConfig, port int) error {
	mac, err := net.ParseMAC(wol.MAC)
	if err != nil {
		return fmt.Errorf("invalid wake-on-lan mac (%s): %w", wol.MAC, err)
	}

	packet, err := buildMagicPacket(mac)
	if err != nil {
		return err
	}

	targetIP := wol.Broadcast
	if targetIP == "" {
		targetIP = wolDefaultBroadcastAddr
	}

	addr := &net.UDPAddr{
		IP:   net.ParseIP(targetIP),
		Port: port,
	}
	if addr.IP == nil {
		return fmt.Errorf("invalid wake-on-lan broadcast address (%s)", targetIP)
	}

	conn, err := net.ListenUDP("udp4", nil)
	if err != nil {
		return err
	}
	defer conn.Close()

	if err := enableUDPSocketBroadcast(conn); err != nil {
		return err
	}

	_, err = conn.WriteToUDP(packet, addr)
	return err
}

func enableUDPSocketBroadcast(conn *net.UDPConn) error {
	raw, err := conn.SyscallConn()
	if err != nil {
		return err
	}

	var controlErr error
	if err := raw.Control(func(fd uintptr) {
		controlErr = syscall.SetsockoptInt(int(fd), syscall.SOL_SOCKET, syscall.SO_BROADCAST, 1)
	}); err != nil {
		return err
	}

	return controlErr
}

func probeTCPReachability(ctx context.Context, target *url.URL) error {
	host := target.Hostname()
	if host == "" {
		return fmt.Errorf("peer proxy target is missing a host")
	}

	port := target.Port()
	if port == "" {
		if strings.EqualFold(target.Scheme, "https") {
			port = "443"
		} else {
			port = "80"
		}
	}

	dialer := &net.Dialer{}
	conn, err := dialer.DialContext(ctx, "tcp", net.JoinHostPort(host, port))
	if err != nil {
		return err
	}
	return conn.Close()
}
