from fastapi import FastAPI
import base64
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


origins = ["*"]


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)





class Item(BaseModel):
    img: str



@app.post("/")
async def root(item :Item):
    print(item.img)

    return "done"