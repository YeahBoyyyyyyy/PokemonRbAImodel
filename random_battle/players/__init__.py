"""Random Battle poke-env players."""

__all__ = ["RbModelPlayer", "RbHybridPlayer"]


def __getattr__(name: str):
    if name == "RbModelPlayer":
        from random_battle.players.rb_model_player import RbModelPlayer

        return RbModelPlayer
    if name == "RbHybridPlayer":
        from random_battle.players.rb_hybrid_player import RbHybridPlayer

        return RbHybridPlayer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
