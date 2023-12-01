# Compile Instructions
```shell

$ git clone --recursive https://github.com/tuero/demo_search.git

# Enter the repository
$ cd demo_search

# Compile
$ cmake . --preset=ubuntu22-gcc-release
$ cmake --build --preset=ubuntu22-gcc-release -- -j32

# Compile python bindings
cmake --build --preset=ubuntu22-gcc-release --target=_hptspy -- -j32
```

# Installing as python bindings
```shell
git clone https://github.com/tuero/demo_search.git
python -m pip install ./demo_search 
```