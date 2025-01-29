if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--max_len", type=int, default=4096)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--checkpoint", type=str, default=None, required=True)
    parser.add_argument("--lr", type=float, default=5e-5)
