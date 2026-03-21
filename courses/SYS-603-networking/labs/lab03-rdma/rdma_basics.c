/*
 * Educational stub — real RDMA uses libibverbs + rdma_cm.
 * Recommended: Linux `rdma-core` examples and Mellanox RDMA Awareness docs.
 *
 * Typical flow (conceptual):
 * 1) ibv_get_device_list / open device context
 * 2) alloc PD, create QP, register MR, exchange connection info with peer
 * 3) post RDMA WRITE/READ or SEND/RECV
 *
 * Do not run this file as-is; it is intentionally incomplete.
 */
#include <stdio.h>
int main(void) {
  puts("See perftest (ib_send_lat / ib_write_lat) for runnable verbs examples.");
  return 0;
}
