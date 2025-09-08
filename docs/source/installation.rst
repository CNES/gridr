Installation
============

The GridR package relies on an internal Rust library (_libgrid) that may not be available for your system configuration. While most users will find the standard installation process straightforward, some may need to follow an advanced installation path to handle this compilation step.


Easy installation
-----------------

The computationally intensive core methods of GridR are implemented in Rust and exposed to Python through PyO3 bindings. This complicates installation. Pre-compiled binary distributions (wheels) containing the internal _libgridr library are available on the Python Package Index and can be installed using pip.

.. code-block:: bash

    pip install gridr

Prebuilt packages are available for x86_64 platforms running a Linux operating system with GLIBC ≥ 2.28. These packages include a precompiled _libgridr Rust library that is not optimized for specific CPU architectures (e.g., SIMD features). If no binary package is available for your system, or if you wish to optimize the _libgridr library for your native CPU architecture, please refer to the advanced installation instructions below.


Advanced installation
---------------------

Once you have a Rust development environment installed (rustc and cargo) on your system (installation instructions are available at https://www.rust-lang.org/tools/install ), you can build and install GridR using setuptools and pip.

First, perform the build step from the root of the source code directory.

.. code-block:: bash

    python -m build

The build process will compile the Rust library using Cargo, Rust's package manager (see https://doc.rust-lang.org/beta/cargo/index.html for more details). Cargo uses the Cargo.toml manifest file (located at ``rust/gridr/Cargo.toml`` from the code root directory) to obtain project configuration, including dependencies.

While you can configure some build options in this file, certain features require setting specific RUSTFLAGS environment variables (see https://doc.rust-lang.org/beta/cargo/reference/environment-variables.html for available options).

You can optimize the library for your specific CPU architecture using the flag `target-cpu=native`.

.. code-block:: bash

    RUSTFLAGS="-C target-cpu=native" python -m build


.. note::

   Building the Rust library requires downloading its dependencies from `crates.io <https://crates.io/>`_ using Cargo.
   This is handled automatically when you run the build command, but requires an internet connection and proper network access to crates.io.
   
   If you're behind a corporate firewall or proxy, you may need to configure Cargo to use it by setting the `HTTP_PROXY` and `HTTPS_PROXY` environment variables.


Once the build completes successfully, you can install the generated wheel using pip.

.. code-block:: bash

   # Replace X.Y.Z and * with the corresponding versions available in your local dist directory
   pip install dist/gridr-X.Y.Z-*.whl
