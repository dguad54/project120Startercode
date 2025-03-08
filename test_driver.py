import torch
import numpy as np
import fastAttention
import argparse
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

def pytorch_vanilla_attention(Q, K, V):
  warmup = 10
  niters = 20
  Q = Q.unsqueeze(0).unsqueeze(1)
  K = K.unsqueeze(0).unsqueeze(1)
  V = V.unsqueeze(0).unsqueeze(1)
  start = torch.cuda.Event(enable_timing=True)
  end = torch.cuda.Event(enable_timing=True)
  with sdpa_kernel(backends=[SDPBackend.MATH]):
    for _ in range(warmup):
      ref_attn = F.scaled_dot_product_attention(Q, K, V, dropout_p=0.0, scale=1.0)
    start.record()
    for _ in range(niters):
      ref_attn = F.scaled_dot_product_attention(Q, K, V, dropout_p=0.0, scale=1.0)
    end.record()
  end.synchronize()
  print(f"Vanilla attention time: {start.elapsed_time(end)/10} ms")
  return ref_attn

def main(args):
  embed_dim = args.embed_dim
  seq_len = args.seq_len
  Q = torch.randn(args.seq_len, args.embed_dim,
                  dtype=torch.float32, device="cuda")
  K = torch.randn(args.seq_len, args.embed_dim,
                  dtype=torch.float32, device="cuda")
  V = torch.randn(args.seq_len, args.embed_dim,
                  dtype=torch.float32, device="cuda")

  ref_attn = pytorch_vanilla_attention(Q, K, V).squeeze(0).squeeze(0)

  # my_attn = fastAttention.naive_attention(Q, K, V)
  # assert torch.allclose(ref_attn, my_attn), "Attention is incorrect"

if __name__ == "__main__":
  # test our transpose kernel
  # only for demonstration purposes
  Q = torch.randn(50, 50, device="cuda")
  QT = fastAttention.naive_transpose(Q)
  QT_ref = Q.T
  assert torch.allclose(QT, QT_ref), "Transpose kernel is incorrect"

  parser = argparse.ArgumentParser()
  parser.add_argument("--embed_dim","-e", type=int, default=128)
  parser.add_argument("--seq_len","-s", type=int, default=1024)
  args = parser.parse_args()

  main(args)
