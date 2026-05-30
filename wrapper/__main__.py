import uvicorn

uvicorn.run("wrapper.app:app", host="0.0.0.0", port=8080, reload=False)
