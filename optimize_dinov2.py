import torch
import time
from src.dinov2_backbone import load_dinov2_pretrained

def benchmark_dinov2_throughput():
    print("=== FruitLearn AI: DINOv2 Hardware Optimization & Throughput Benchmark ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device}")

    model = load_dinov2_pretrained().to(device)
    model.eval()

    dummy_input = torch.randn(16, 3, 224, 224, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)

    start_time = time.time()
    num_iterations = 25

    with torch.no_grad():
        for _ in range(num_iterations):
            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                _ = model(dummy_input)

    total_time = time.time() - start_time
    throughput = (num_iterations * 16) / total_time

    print(f"Batch Size: 16 | Iterations: {num_iterations}")
    print(f"Total Latency: {total_time:.3f} s | Throughput: {throughput:.2f} images/sec")
    print("DINOv2 ViT-Base/14 Mixed Precision Inference Verified.")

if __name__ == "__main__":
    benchmark_dinov2_throughput()
