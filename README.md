# AI spam detection

This is an example project on how to use tch-rs in Rust for a task of detecting
spam messages.

Ths project is split into two parts: the library crate that describes the model
(the root directory) and the binary crate for training, inference and dataset
preparation (the `training` directory).

The project features an end-to-end process of making a model from dataset
preparation down to the inference.

## torch library

The project relies on the `tch-rs` crate, which relies on the `torch` library.
In order for everything to work as expected, you need to download and unpack the
libtorch (cxx11 abi) archive, then set export the environment variables
`LIBTORCH=/path/to/libtorch-2.3.0` and
`LD_LIBRARY_PATH=/path/to/libtorch-2.3.0/lib`.

If you use "Visual Studio Code" with "rust-analyzer" on Linux this
`.vscode/settings.json` file can make your life a bit easier:

```json
{
    "terminal.integrated.env.linux": {
        "LIBTORCH": "/path/to/libtorch-2.3.0",
        "LD_LIBRARY_PATH": "/path/to/libtorch-2.3.0/lib"
    },
    "rust-analyzer.runnables.extraEnv": {
        "LIBTORCH": "/path/to/libtorch-2.3.0",
        "LD_LIBRARY_PATH": "/path/to/libtorch-2.3.0/lib"
    },
    "rust-analyzer.cargo.extraEnv": {
        "LIBTORCH": "/path/to/libtorch-2.3.0",
        "LD_LIBRARY_PATH": "/path/to/libtorch-2.3.0/lib"
    }
}
```

## ONNX conversion

When the model is trained, you can use the `convert_onnx.py` python application
to convert the trained model into the `ONNX` format. The script requires the
`torch`, `safetensors` and `transformer` libraries to be installed.
