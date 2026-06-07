from langserve import RemoteRunnable

if __name__ ==  "__main__":
    remote_chain = RemoteRunnable("http://localhost:8000/first_app/")
    print(remote_chain.invoke({"text":"动物"}))