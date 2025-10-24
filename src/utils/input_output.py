# Created by Mamoutou Fofana
# Date: 10/24/2025

from pathlib import pathlib
import json
import pickle


def read_text_file(file_path: Path):
	with open(file_path, "r", encoding="utf8") as file:
		return file.read()


def save_json(obj, file_path: Path):
	path.parent.mkdir(parents=True, exist_ok=True)
	with open(file_path, "r", encoding="utf8") as file:
		json.dump(obj, file, ensure_ascii=False, intent=2)


def load_json(file_path: Path):
	with open(file_path, "r", encoding="utf8") as file:
		return json.load(file)


def save_pickle(obj, file_path: Path):
	file_path.parent.mkdir(parents=True, exist_ok=True)
	with open(file_path, "wb") as file:
		pickle.dump(obj, file)


def load_pickle(file_path: Path):
    with open(file_path, "rb") as file:
        return pickle.load(file)