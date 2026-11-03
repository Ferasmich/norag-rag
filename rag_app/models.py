from pydantic import BaseModel, ConfigDict, Field, model_validator


class Settings(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    chunk_size: int = Field(default=800, ge=100, le=1000)
    chunk_overlap: int = Field(default=80, ge=0)
    top_k: int = Field(default=5, ge=1, le=20)

    @model_validator(mode='after')
    def valid_overlap(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError('Overlap must be smaller than chunk size.')
        return self


class Selection(BaseModel):
    document_ids: list[str] = Field(min_length=1, max_length=100)
    settings: Settings = Field(default_factory=Settings)


class Question(Selection):
    question: str = Field(min_length=1, max_length=2000)

    @model_validator(mode='after')
    def nonblank(self):
        if not self.question.strip():
            raise ValueError('Enter a question.')
        return self


class GeneratedAnswer(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    answer: str = Field(min_length=1, max_length=12000)
    answerable: bool
    citations: list[str] = Field(max_length=20)

    @model_validator(mode='after')
    def consistent(self):
        if not self.answer.strip():
            raise ValueError('Answer cannot be blank.')
        if self.answerable != bool(self.citations):
            raise ValueError('An answer needs citations; an abstention must not cite evidence.')
        if len(set(self.citations)) != len(self.citations):
            raise ValueError('Duplicate citations.')
        return self


class WordsRequest(Selection):
    question: str = Field(default='', max_length=2000)
    word: str = Field(default='', max_length=200)


class EvaluationRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    corpus: str = Field(pattern='^(kestrel|vellora)$')
    settings: Settings = Field(default_factory=Settings)
    answers: bool = False
    no_context: bool = False


class GradeRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    question_id: str = Field(min_length=1, max_length=80)
    grade: str | None = Field(default=None, pattern='^(correct|partial|wrong|unsupported)$')
    notes: str = Field(min_length=1, max_length=3000)
    reviewer: str = Field(min_length=1, max_length=100)
    no_context_correct: bool | None = None

    @model_validator(mode='after')
    def has_review(self):
        if self.grade is None and self.no_context_correct is None:
            raise ValueError('Provide an answer grade or a no-context correctness flag.')
        return self
