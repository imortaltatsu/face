"""
GPU Configuration for TensorFlow Training

Supports:
1. Local GPU (CUDA/Metal)
2. Remote GPU via TensorFlow Serving
3. Multi-GPU training
4. Mixed precision training for faster performance
"""

import tensorflow as tf
import os


def configure_gpu(use_mixed_precision=True, memory_growth=True):
    """
    Configure GPU settings for optimal training
    
    Args:
        use_mixed_precision: Enable mixed precision (FP16) for faster training
        memory_growth: Enable memory growth to avoid OOM errors
    """
    print("\n🖥️  GPU Configuration")
    print("=" * 60)
    
    # Check available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print(f"✅ Found {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
        
        try:
            # Enable memory growth
            if memory_growth:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("✅ Memory growth enabled")
            
            # Set visible devices (use all GPUs)
            tf.config.set_visible_devices(gpus, 'GPU')
            
            # Enable mixed precision for faster training
            if use_mixed_precision:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                print("✅ Mixed precision (FP16) enabled")
                print("   → ~2x faster training on modern GPUs")
            
        except RuntimeError as e:
            print(f"⚠️  GPU configuration error: {e}")
    else:
        print("⚠️  No GPU found, using CPU")
        print("   Training will be slower")
    
    print("=" * 60)


def configure_remote_gpu(remote_server_url):
    """
    Configure remote GPU training via TensorFlow Serving
    
    Args:
        remote_server_url: URL of remote TensorFlow server (e.g., 'grpc://192.168.1.100:8500')
    
    Note: This requires TensorFlow Serving to be running on the remote machine
    """
    print(f"\n🌐 Configuring remote GPU: {remote_server_url}")
    
    # Set TensorFlow to use remote server
    os.environ['TF_CONFIG'] = f'{{"cluster": {{"worker": ["{remote_server_url}"]}}}}'
    
    print("✅ Remote GPU configured")
    print("   Make sure TensorFlow Serving is running on the remote machine")


def configure_multi_gpu_strategy():
    """
    Configure multi-GPU training strategy optimized for 8x A100
    
    Returns:
        tf.distribute.Strategy for multi-GPU training
    """
    gpus = tf.config.list_physical_devices('GPU')
    
    if len(gpus) > 1:
        print(f"\n🚀 Multi-GPU Training: {len(gpus)} GPUs")
        
        # Use NCCL for fast multi-GPU communication on A100s
        strategy = tf.distribute.MirroredStrategy(
            cross_device_ops=tf.distribute.NcclAllReduce()
        )
        
        print(f"✅ MirroredStrategy with NCCL enabled")
        print(f"   → Training will use all {len(gpus)} GPUs")
        print(f"   → Effective batch size: {64 * len(gpus)} (64 per GPU)")
        print(f"   → Using NCCL for fast GPU communication")
        return strategy
    elif len(gpus) == 1:
        print("\n💻 Single GPU Training")
        return tf.distribute.get_strategy()  # Default strategy
    else:
        return tf.distribute.get_strategy()


def setup_gpu_strategy(base_batch_size=16):
    """
    Setup GPU strategy and calculate total batch size
    
    Args:
        base_batch_size: Batch size per GPU
        
    Returns:
        (strategy, num_gpus, total_batch_size)
    """
    gpus = tf.config.list_physical_devices('GPU')
    num_gpus = len(gpus)
    
    if num_gpus > 1:
        print(f"\n🚀 Multi-GPU Training: {num_gpus} GPUs")
        # Use NCCL for fast multi-GPU communication
        strategy = tf.distribute.MirroredStrategy(
            cross_device_ops=tf.distribute.NcclAllReduce()
        )
        print(f"✅ MirroredStrategy enabled")
    elif num_gpus == 1:
        print("\n💻 Single GPU Training")
        strategy = tf.distribute.get_strategy()
    else:
        print("\n⚠️  CPU Training (no GPU found)")
        strategy = tf.distribute.get_strategy()
        num_gpus = 1  # Treat CPU as 1 device for batch calculation
    
    total_batch_size = base_batch_size * num_gpus
    
    return strategy, num_gpus, total_batch_size


def get_optimal_batch_size(image_size, gpu_memory_gb=40):
    """
    Calculate optimal batch size based on GPU memory
    
    Args:
        image_size: Image size (e.g., 544)
        gpu_memory_gb: GPU memory in GB (default 40 for A100)
    
    Returns:
        Recommended batch size
    """
    # Base batch sizes optimized for A100 40GB GPUs
    if image_size <= 224:
        base_batch = 256
    elif image_size <= 384:
        base_batch = 128
    elif image_size <= 544:
        base_batch = 64  # Increased from 16 for A100
    else:
        base_batch = 32
    
    # Scale by GPU memory
    memory_factor = gpu_memory_gb / 40  # Assume 40GB A100 as baseline
    optimal_batch = int(base_batch * memory_factor)
    
    return max(16, optimal_batch)  # Minimum batch size of 16


# Example usage configurations
CONFIGS = {
    'local_gpu': {
        'name': 'Local GPU (CUDA/Metal)',
        'description': 'Use local GPU with mixed precision',
        'setup': lambda: configure_gpu(use_mixed_precision=True, memory_growth=True)
    },
    'local_cpu': {
        'name': 'Local CPU',
        'description': 'CPU-only training (slower)',
        'setup': lambda: configure_gpu(use_mixed_precision=False, memory_growth=False)
    },
    'multi_gpu': {
        'name': 'Multi-GPU',
        'description': 'Distributed training across multiple GPUs',
        'setup': lambda: configure_multi_gpu_strategy()
    },
    'remote_gpu': {
        'name': 'Remote GPU',
        'description': 'Use remote GPU via TensorFlow Serving',
        'setup': lambda url: configure_remote_gpu(url)
    }
}


def show_gpu_info():
    """Display detailed GPU information"""
    print("\n📊 GPU Information")
    print("=" * 60)
    
    # Physical devices
    gpus = tf.config.list_physical_devices('GPU')
    cpus = tf.config.list_physical_devices('CPU')
    
    print(f"GPUs: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"  {i}: {gpu}")
    
    print(f"\nCPUs: {len(cpus)}")
    
    # Check if GPU is actually being used
    print(f"\nGPU Available: {tf.test.is_gpu_available()}")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    
    # Current device placement
    with tf.device('/GPU:0' if gpus else '/CPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
        print(f"\nTest computation device: {c.device}")
    
    print("=" * 60)


if __name__ == "__main__":
    # Show GPU info
    show_gpu_info()
    
    # Configure for training
    print("\n🎯 Recommended Configuration:")
    
    gpus = tf.config.list_physical_devices('GPU')
    
    if len(gpus) > 1:
        print("   → Multi-GPU training")
        strategy = configure_multi_gpu_strategy()
    elif len(gpus) == 1:
        print("   → Single GPU with mixed precision")
        configure_gpu(use_mixed_precision=True, memory_growth=True)
    else:
        print("   → CPU training")
        print("   ⚠️  Consider using Google Colab or remote GPU for faster training")
    
    # Show optimal batch size
    optimal_batch = get_optimal_batch_size(544, gpu_memory_gb=8)
    print(f"\n📦 Optimal batch size for 544x544 images: {optimal_batch}")
