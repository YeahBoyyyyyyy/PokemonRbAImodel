"""Random Battle (gen9randombattle) branch defaults."""

from pathlib import Path

from common.project_paths import PROJECT_ROOT, RB_DIR
from poke_env.ps_client.server_configuration import ServerConfiguration

FORMAT_ID = "gen9randombattle"

# poke-env default LocalhostServerConfiguration still authenticates against play.pokemonshowdown.com
LOCAL_SERVER_CONFIGURATION = ServerConfiguration(
    "ws://localhost:8000/showdown/websocket",
    "http://localhost:8000/action.php?",
)
FORMAT_NAME = ""
MIN_ELO_DEFAULT = 1400

DATA_DIR = RB_DIR / "data"
ARTIFACTS_DIR = RB_DIR / "artifacts"
RB_SET_DEX_PATH = DATA_DIR / "rb_set_dex.json"
MOVE_VOCAB_PATH = ARTIFACTS_DIR / "move_vocab.json"
MULTIHEAD_MODEL_DIR = ARTIFACTS_DIR / "multihead_model"
ACTION_MODEL_DIR = ARTIFACTS_DIR / "action_model"
MOVE_MODEL_DIR = ARTIFACTS_DIR / "move_model"
WINRATE_MODEL_DIR = ARTIFACTS_DIR / "winrate_model"
REPLAYS_DIR = ARTIFACTS_DIR / "replays"
SESSION_DIR = ARTIFACTS_DIR / "sessions"

ACTION_CHUNKS_DIR = PROJECT_ROOT / "random_battle" / "chunks" / "action"
ACTION_CHUNKS_HOLIDAYOUGI_DIR = PROJECT_ROOT / "random_battle" / "chunks" / "action_holidayougi"
WINRATE_CHUNKS_DIR = PROJECT_ROOT / "random_battle" / "chunks" / "winrate"
