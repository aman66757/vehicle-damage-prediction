from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def home():
    return {
        "message": "Hello! FastAPI is working!"
    }


@app.get("/about")
def about():
    return {
        "project": "Vehicle Damage Classification",
        "status": "Running"
    }