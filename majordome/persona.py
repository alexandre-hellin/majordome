from pathlib import Path
from datetime import datetime
from majordome.config import config
import tomllib

PERSONAS_DIR = Path(__file__).parent.parent / "personas"
DEFAULT_PERSONA = config.get("persona", {}).get("name", "default")
WEEKDAY = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]

_persona: Persona | None = None


class Persona:
    def __init__(self, name: str = DEFAULT_PERSONA):
        self.path = PERSONAS_DIR / name
        self.audio = str(self.path / "voice.wav")
        self.config = self._load_config()

    def _load_config(self) -> dict:
        toml_path = self.path / "persona.toml"
        if toml_path.exists():
            with open(toml_path, "rb") as f:
                return tomllib.load(f)
        return {}

    @property
    def voice_transcription(self) -> str | None:
        return self.config.get("voice", {}).get("transcription", None)

    @property
    def display_name(self) -> str:
        return self.config.get("persona", {}).get("name", self.path.name)

    def render_prompt(self, extra: dict | None = None) -> str:
        template = self.config.get("prompt", {}).get("system", "")
        context = _build_context()
        context |= self.config.get("persona", {})
        if extra:
            context |= extra
        return template.format_map(context)


def preload():
    """Preload the Persona at startup."""
    _init_persona()


def get_persona() -> Persona:
    """Singleton accessor for the Persona."""
    if _persona is None:
        raise RuntimeError("Persona not initialized. Call preload() first.")
    return _persona


def _build_context() -> dict:
    now = datetime.now()
    return {
        "time": now.strftime("%H:%M"),
        "date": now.strftime("%d/%m/%Y"),
        "weekday": WEEKDAY[now.weekday()],
    }


def _init_persona():
    """Initialize the persona if not already loaded."""
    global _persona
    if _persona is None:
        _persona = Persona()
