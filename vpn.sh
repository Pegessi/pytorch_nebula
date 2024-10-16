export no_proxy="localhost, 127.0.0.1, ::1"

# 定义一个通用的函数，用于设置代理和git代理配置
export_proxy() {
    export http_proxy=$proxy
    export https_proxy=$proxy
    git config --global http.proxy $proxy
    git config --global https.proxy $proxy
}

test_proxy() {
    echo "Testing proxy $proxy..."
    if curl -s --head --request GET www.google.com --max-time 5 > /dev/null; then
        echo "Proxy $proxy is working."
    else
        echo "Proxy $proxy is not working."
        # 如果代理不可用，可以选择清除代理设置
        unset http_proxy
        unset https_proxy
        git config --global --unset http.proxy
        git config --global --unset https.proxy
    fi
}

unset_proxy() {
    unset http_proxy
    unset https_proxy
    git config --global --unset http.proxy
    git config --global --unset https.proxy
}

# 通过定义不同的函数来设置不同的代理
loyer_proxy() {
    export proxy="http://10.18.86.222:51664"
    export_proxy
    test_proxy
}

zehua_proxy() {
    export proxy="http://10.18.86.26:7890"
    export_proxy
    test_proxy
}

kangyu_proxy() {
    export proxy="http://10.18.86.22:10811"
    export_proxy
    test_proxy
}

yunfei_proxy() {
    export proxy="http://10.18.95.26:7890"
    export_proxy
    test_proxy
}

zehua_proxy