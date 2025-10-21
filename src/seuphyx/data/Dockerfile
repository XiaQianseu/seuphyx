# 基于 Julia 官方镜像
FROM julia:1.10-bullseye

# 1) 安装 Python & pip
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*

# 2) Python 依赖（含 PySR；Streamlit 用于网页）
ENV PIP_NO_CACHE_DIR=1
RUN python3 -m pip install --upgrade pip setuptools wheel
RUN python3 -m pip install --no-cache-dir \
    numpy pandas matplotlib scipy scikit-learn sympy pysr joblib streamlit

# 3) 创建普通用户，设置 HOME 和 Julia depot（关键，避免写到 /.julia）
RUN useradd -m -u 1000 appuser
ENV HOME=/home/appuser
ENV JULIA_DEPOT_PATH=/home/appuser/.julia
ENV JULIA_NUM_THREADS=auto
ENV MPLBACKEND=Agg
ENV PYTHONUNBUFFERED=1

# 4) 拷贝代码并授权
WORKDIR /app
COPY . /app
RUN chown -R appuser:appuser /app /home/appuser

# 5) 切到普通用户，预装/预编译 Julia 包（写入 $JULIA_DEPOT_PATH）
USER appuser
RUN julia -e 'using Pkg; Pkg.add(["SymbolicRegression","PythonCall","PyCall"]); Pkg.precompile()'

# 6) 监听 7860（HF 要求），以 Streamlit 形式启动 app.py
EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]

