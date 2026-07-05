def test_embedding():
    from langchain_community.embeddings.dashscope import DashScopeEmbeddings
    embeddings_model = DashScopeEmbeddings()
    embeddings = embeddings_model.embed_documents(
        [
            """《崂山道士》：这是一首rap歌曲，
            有句歌词是想学功夫修炼仙术，
            先征服这条山路，
            我在三清观里录歌，
            旋律和韵脚兼顾，看我点石成金，把黑的说成白，想抓我的随便来；
            这句歌词生动的描绘了创作者急于想要成功的心态。"
            """,
            """《黑怕不怕黑》：这是一首rap歌曲，
            有句歌词是晚上睡不着觉，
            白天起不来，
            我皮肤不够黑，也不够白，他们总觉得黄种人的说唱一点也不独特；这句歌词表达了
            创作者对于中国本土说唱不受欢迎的想法。"""
        ]
    )
    embedded_query = embeddings_model.embed_query("我想听一首关于急于求成的歌曲")
    print(len(embeddings),len(embeddings[0]),len(embedded_query))
    
test_embedding()