from scapy.all import sniff
from sklearn.preprocessing import StandardScaler  # (Currently unused, but ok to import)

def packet_callback(packet):
    print(packet.summary())  # This prints a short summary of each packet

sniff(
    iface="\\Device\\NPF_{4E568DBF-30BA-4120-B489-E89D6FFEECCD}",  # <-- Replace with actual interface string
    filter="ip", 
    prn=packet_callback, 
    store=0
)