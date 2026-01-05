# uv python 下载
``` shell
export UV_PYTHON_INSTALL_MIRROR="https://ghfast.top/https://github.com/astral-sh/python-build-standalone/releases/download"
```
# uv proxy

``` toml
[[tool.uv.index]]
url = "https://nexus.infra.agibot.com/repository/pypi-proxy/simple"
default = true
```
or 
``` toml
[[tool.uv.index]]
url = "https://pypi.tuna.tsinghua.edu.cn/simple"
default = true
```