# test_avgpool.py

import torch
import torch.nn.functional as F
from cublade.bindings.pooling.avgpool1d import avg_pool_1d

def test_avg_pool_1d_random():
    # Rastgele test
    L = 127
    kernel_size = 5
    stride = 3

    x = torch.randn(L, device="cuda", dtype=torch.float32)
    y = avg_pool_1d(x, kernel_size, stride)

    # PyTorch referans: [N,C,L] -> (1,1,L)
    y_ref = F.avg_pool1d(x.view(1,1,-1), kernel_size, stride).view(-1)

    if not torch.allclose(y, y_ref, atol=1e-6):
        print("❌ Random test FAILED")
        print("Got:     ", y)
        print("Expected:", y_ref)
        return False
    print("✅ Random test passed")
    return True

def test_avg_pool_1d_known():
    # Bilinen girdi ile test
    x = torch.arange(10, device="cuda", dtype=torch.float32)  # [0,1,2,...,9]
    kernel_size = 3
    stride = 2

    y = avg_pool_1d(x, kernel_size, stride)
    # manuel hesap: [(0+1+2)/3, (2+3+4)/3, ...]
    expected = torch.tensor([1.0, 3.0, 5.0, 7.0], device="cuda")

    if not torch.equal(y, expected):
        print("❌ Known-input test FAILED")
        print("Got:     ", y)
        print("Expected:", expected)
        return False
    print("✅ Known-input test passed")
    return True

if __name__ == "__main__":
    print("Running avg_pool_1d tests...")
    ok1 = test_avg_pool_1d_random()
    ok2 = test_avg_pool_1d_known()
    if ok1 and ok2:
        print("\n🎉 All tests passed!")
        exit(0)
    else:
        print("\n❗ Some tests failed.")
        exit(1)
