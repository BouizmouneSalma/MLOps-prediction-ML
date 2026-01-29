from fastapi import FastAPI

app = FastAPI()


@app.post("/")
def test():
    
    return {"statut": "server is healthy"}

