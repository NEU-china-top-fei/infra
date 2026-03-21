# RoCEv2 vs InfiniBand (cheat sheet)

| Topic | InfiniBand | RoCEv2 |
|-------|------------|--------|
| L2/L3 | IB link layer | Ethernet + IP + UDP |
| Lossless | native credit-based flow control | PFC + (sometimes) ECN |
| Deployment | common in HPC fabrics | common in cloud / enterprise Ethernet |
| Multipath | different routing idioms | ECMP hashing → watch tail latency |

Interview angles: PFC deadlock domains, head-of-line blocking, need for congestion control tuning on large AI clusters.
