# 1. Nsys 中如何看gpu kernel？
| 信息 | 现在问什么 |
|---|---|
| **Name** | 这是哪个 kernel？ |
| **Begin / Duration** | 真正在 GPU 上跑多久？ |
| **Stream** | 在哪个 stream？ |
| **Correlation ID** | 是哪个 Host launch 发出来的？ |
| **Latency** | launch 后等多久才开始？ |
| **grid / block** | GPU 工作怎么切的？ |
| **registers/thread** | 每线程资源压力如何？ |
| **shared memory / occupancy** | block 驻留可能受到什么限制？ |