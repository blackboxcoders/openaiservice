from google.cloud import storage

def download_embedding():
    #bucket_name = "vectorstore_chatbot_test"
    bucket_name = "youngliving_vectorestore"
    path = "local/whatssap-chatbot-openai-e5e3a1b6539f.json"
    store_client = storage.Client.from_service_account_json(path)

    bucket = storage.Bucket(store_client, bucket_name)

    blob_1 = bucket.blob("PdfVectorStore/faiss_index/index.faiss")
    blob_2 = bucket.blob("PdfVectorStore/faiss_index/index.pkl")

    blob_1.download_to_filename("vectorStore/faiss_index/index.faiss")
    blob_2.download_to_filename("vectorStore/faiss_index/index.pkl")

download_embedding()
