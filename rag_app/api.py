import logging

from fastapi import FastAPI, File, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .models import Question, Selection
from .service import MAX_BYTES, ROOT, PipelineError, RagService


def create_app(service=None):
    app = FastAPI(title='Evidence / RAG V1')
    app.state.rag = service or RagService()

    @app.exception_handler(PipelineError)
    async def pipeline_error(request, exc):
        return JSONResponse(status_code=exc.status, content={'error': str(exc)})

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        return JSONResponse(status_code=422, content={'error': 'Invalid input. Select documents, enter a question and check retrieval settings.'})

    @app.exception_handler(Exception)
    async def unexpected(request, exc):
        logging.getLogger(__name__).exception('Unhandled request error')
        return JSONResponse(status_code=500, content={'error': 'Operation failed. Check server logs and retry.'})

    @app.get('/api/documents')
    def documents():
        with app.state.rag.lock:
            return app.state.rag.list_documents()

    @app.post('/api/documents')
    def upload(file: UploadFile = File(...)):
        return app.state.rag.upload(file.filename or '', file.file.read(MAX_BYTES + 1))

    @app.delete('/api/documents/{doc_id}')
    def delete(doc_id: str):
        app.state.rag.delete(doc_id)
        return {'deleted': doc_id}

    @app.post('/api/index')
    def index(body: Selection):
        return app.state.rag.index(body.document_ids, body.settings)

    @app.post('/api/ask')
    def ask(body: Question):
        return app.state.rag.ask(body.question, body.document_ids, body.settings)

    @app.get('/api/evaluation')
    def evaluation():
        import json
        path = ROOT / 'evaluation' / 'baseline.json'
        return json.loads(path.read_text()) if path.exists() else {'status': 'Run python -m evaluation.run first.'}

    app.mount('/', StaticFiles(directory=ROOT / 'rag_app' / 'static', html=True), name='frontend')
    return app


app = create_app()
