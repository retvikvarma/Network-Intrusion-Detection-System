from scapy.all import IP, TCP, send
import time

print("🚨 Sending simulated malicious packets...")

# Configuration
target_ip = "127.0.0.1"  # Replace with the IP your NIDS is monitoring
target_port = 80         # Common HTTP port
num_packets = 10         # Number of SYN packets

for i in range(num_packets):
    packet = IP(dst=target_ip) / TCP(dport=target_port, sport=10000 + i, flags="S")
    send(packet, verbose=False)
    print(f"📤 Sent packet #{i+1}")
    time.sleep(0.3)

print("✅ All packets sent.")