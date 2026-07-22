import csv
import matplotlib.pyplot as plt
import os

csv_path = "outputs/remote_sensing_scenario/20260612_171122/figures_600slots/end_to_end_delay_energy_by_slot_600slots.csv"
if not os.path.exists(csv_path):
    print(f"File not found: {csv_path}")
    exit(1)

slots = []
delays = []
with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            slot = int(row["time_slot"])
            # Calculate single packet delay (divide by 16.0)
            delay = float(row["avg_delay_ms"]) / 16.0
            slots.append(slot)
            delays.append(delay)
        except ValueError:
            pass

plt.figure(figsize=(10, 5))
plt.plot(slots, delays, color="#1f77b4", linewidth=1.5)
plt.title("Single Packet Delay per Time Slot")
plt.xlabel("Time Slot")
plt.ylabel("Delay (ms)")
plt.grid(True, linestyle="--", alpha=0.5)

out_path = "outputs/remote_sensing_scenario/20260612_171122/figures_600slots/single_packet_delay.png"
plt.savefig(out_path, dpi=150)
print(f"Saved to {out_path}")
