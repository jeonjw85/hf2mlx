from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from typing_extensions import override


class Hf2mlxError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class GatedModelError(Hf2mlxError):
    model: str

    @override
    def __str__(self) -> str:
        return "Error: model is gated. Set HF_TOKEN and retry."


@dataclass(frozen=True, slots=True)
class ModelNotFoundError(Hf2mlxError):
    model: str

    @override
    def __str__(self) -> str:
        return f"Error: model not found: {self.model}"


@dataclass(frozen=True, slots=True)
class InvalidModelIdError(Hf2mlxError):
    model: str

    @override
    def __str__(self) -> str:
        return f"Error: invalid model id: {self.model}"


@dataclass(frozen=True, slots=True)
class LocalModelNotFoundError(Hf2mlxError):
    path: Path

    @override
    def __str__(self) -> str:
        return f"Error: local model path does not exist: {self.path}"


@dataclass(frozen=True, slots=True)
class LocalModelNotDirError(Hf2mlxError):
    path: Path

    @override
    def __str__(self) -> str:
        return f"Error: not a model directory: {self.path}"


@dataclass(frozen=True, slots=True)
class OutputExistsError(Hf2mlxError):
    path: Path

    @override
    def __str__(self) -> str:
        return f"Error: output already exists: {self.path}. Use --force to overwrite."


@dataclass(frozen=True, slots=True)
class InsufficientDiskError(Hf2mlxError):
    needed_bytes: int
    available_bytes: int
    needed_label: str
    available_label: str

    @override
    def __str__(self) -> str:
        need = self.needed_label
        have = self.available_label
        return f"Error: insufficient disk space. Need ~{need}, have {have}."


@dataclass(frozen=True, slots=True)
class InsufficientMemoryError(Hf2mlxError):
    needed_label: str
    available_label: str

    @override
    def __str__(self) -> str:
        need = self.needed_label
        have = self.available_label
        return f"Error: insufficient memory. Need ~{need} for weights, have {have}."


@dataclass(frozen=True, slots=True)
class UnsupportedArchitectureError(Hf2mlxError):
    reason: str

    @override
    def __str__(self) -> str:
        return f"Error: unsupported architecture. {self.reason}"


@dataclass(frozen=True, slots=True)
class ConversionFailedError(Hf2mlxError):
    reason: str

    @override
    def __str__(self) -> str:
        return f"Error: conversion failed. {self.reason}"


@dataclass(frozen=True, slots=True)
class NetworkError(Hf2mlxError):
    reason: str

    @override
    def __str__(self) -> str:
        return f"Error: could not reach Hugging Face. {self.reason}"


@dataclass(frozen=True, slots=True)
class HubRequestError(Hf2mlxError):
    model: str
    status: int

    @override
    def __str__(self) -> str:
        return f"Error: Hugging Face request failed ({self.status}) for {self.model}."


@dataclass(frozen=True, slots=True)
class GgufToolsMissingError(Hf2mlxError):
    reason: str

    @override
    def __str__(self) -> str:
        return f"Error: GGUF tools missing. {self.reason}"


@dataclass(frozen=True, slots=True)
class ParamCountUnknownError(Hf2mlxError):
    model: str

    @override
    def __str__(self) -> str:
        return f"Error: could not determine parameter count for {self.model}."


@dataclass(frozen=True, slots=True)
class MlxMissingError(Hf2mlxError):
    @override
    def __str__(self) -> str:
        return "Error: mlx-lm is not installed. Run: uv pip install mlx-lm"


@dataclass(frozen=True, slots=True)
class UnexpectedError(Hf2mlxError):
    reason: str

    @override
    def __str__(self) -> str:
        return f"Error: unexpected failure. {self.reason}"
