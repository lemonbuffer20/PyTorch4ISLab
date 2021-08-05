from build import build_model, build_dataset


def generate(prefix: str,
             generate_length: int,
             data_root: str,
             device="cuda",
             verbose: bool = False,
             checkpoint_path: str = "") -> str:
    if not checkpoint_path:
        raise ValueError("Generation need checkpoint to be loaded")
    # ---------------------------------------------------------------- #
    # Build
    print("-" * 64)
    # ---------------------------------------------------------------- #
    _, _, _, dictionary = build_dataset(data_root)
    vocab_size = len(dictionary)

    model = build_model(vocab_size, verbose=verbose, checkpoint=checkpoint_path)
    model.to(device)

    # ---------------------------------------------------------------- #
    # Generate
    print("-" * 64)
    # ---------------------------------------------------------------- #
    model.eval()
    indices = dictionary.encode(prefix, use_unknown=True)
    generated_indices = model.greedy_generate(indices, generate_length=generate_length)
    postfix = dictionary.decode(generated_indices)
    out = prefix + " " + postfix
    return out


if __name__ == '__main__':
    text = "i am happy because"
    output = generate(text, 100,
                      data_root="ptb",
                      device="cuda",
                      verbose=False,
                      checkpoint_path="result/best.pth")
    print(f"Generate done:\n"
          f"... prefix: {text}\n"
          f"... generated: {output}")
