from redis import Redis


r = Redis(
    host="localhost",
    port=6380,
    decode_responses=True
)


r.set(
    "name",
    "FastAPI"
)


print(
    r.get("name")
)