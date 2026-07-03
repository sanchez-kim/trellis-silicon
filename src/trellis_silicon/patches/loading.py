"""Model-loading patches: conditional checkpoint loading (skip checkpoints a
given pipeline_type never uses) and skip-init (drop the wasteful weight
initialization that load_state_dict immediately overwrites)."""

import os

from .common import TRELLIS_ROOT, read_file, write_file


def patch_base_models_to_load():
    """Let Pipeline.from_pretrained accept an explicit models_to_load list that
    overrides the class-level model_names_to_load, so callers can skip loading
    checkpoints they won't use (Task A — conditional checkpoint loading).
    """
    path = os.path.join(TRELLIS_ROOT, "trellis2/pipelines/base.py")
    src = read_file(path)

    if "models_to_load: Optional[list]" in src:
        print(f"  Already patched: {os.path.relpath(path, TRELLIS_ROOT)}")
        return

    src = src.replace(
        '    def from_pretrained(cls, path: str, config_file: str = "pipeline.json") -> "Pipeline":',
        '    def from_pretrained(cls, path: str, config_file: str = "pipeline.json", models_to_load: Optional[list] = None) -> "Pipeline":',
    )

    src = src.replace(
        "        _models = {}\n"
        "        for k, v in args['models'].items():\n"
        "            if hasattr(cls, 'model_names_to_load') and k not in cls.model_names_to_load:\n"
        "                continue\n",
        "        _models = {}\n"
        "        _allowed = models_to_load if models_to_load is not None else (\n"
        "            cls.model_names_to_load if hasattr(cls, 'model_names_to_load') else None)\n"
        "        for k, v in args['models'].items():\n"
        "            if _allowed is not None and k not in _allowed:\n"
        "                continue\n",
    )
    write_file(path, src)


def patch_pipeline_conditional_load():
    """Thread a pipeline_type through Trellis2ImageTo3DPipeline.from_pretrained
    so only the checkpoints needed for that pipeline_type are loaded (Task A).
    For pipeline_type='512' this skips the two ~2.4GB *_1024 flow models that
    are otherwise loaded but never used.
    """
    path = os.path.join(TRELLIS_ROOT, "trellis2/pipelines/trellis2_image_to_3d.py")
    src = read_file(path)

    if "_models_for_pipeline_type" in src:
        print(f"  Already patched: {os.path.relpath(path, TRELLIS_ROOT)}")
        return

    src = src.replace(
        "    @classmethod\n"
        '    def from_pretrained(cls, path: str, config_file: str = "pipeline.json") -> "Trellis2ImageTo3DPipeline":',
        "    @staticmethod\n"
        "    def _models_for_pipeline_type(pipeline_type: Optional[str]) -> Optional[list]:\n"
        '        """Return the checkpoint names required for a given pipeline_type,\n'
        '        or None (load everything) for an unknown/None type."""\n'
        "        base = [\n"
        "            'sparse_structure_flow_model',\n"
        "            'sparse_structure_decoder',\n"
        "            'shape_slat_decoder',\n"
        "            'tex_slat_decoder',\n"
        "        ]\n"
        "        extra = {\n"
        "            '512': ['shape_slat_flow_model_512', 'tex_slat_flow_model_512'],\n"
        "            '1024': ['shape_slat_flow_model_1024', 'tex_slat_flow_model_1024'],\n"
        "            '1024_cascade': ['shape_slat_flow_model_512', 'shape_slat_flow_model_1024', 'tex_slat_flow_model_1024'],\n"
        "        }\n"
        "        if pipeline_type not in extra:\n"
        "            return None\n"
        "        return base + extra[pipeline_type]\n"
        "\n"
        "    @classmethod\n"
        '    def from_pretrained(cls, path: str, config_file: str = "pipeline.json", pipeline_type: Optional[str] = None) -> "Trellis2ImageTo3DPipeline":',
    )

    src = src.replace(
        "        pipeline = super().from_pretrained(path, config_file)\n",
        "        models_to_load = cls._models_for_pipeline_type(pipeline_type) if pipeline_type is not None else None\n"
        "        pipeline = super().from_pretrained(path, config_file, models_to_load=models_to_load)\n",
    )
    write_file(path, src)


def patch_skip_init_on_load():
    """Skip initialize_weights() when loading a model from a checkpoint.

    models.from_pretrained constructs the module (which runs the model's
    initialize_weights() — xavier/normal init over every parameter) and then
    immediately calls load_state_dict, overwriting all of it. For the 1.3B flow
    models that wasted init costs 40-70s each of pure RNG generation. Skipping
    it is verified bit-identical to the initialized-then-loaded model (the
    checkpoint covers every parameter), so no math changes.

    This runs synchronously before the pipeline is used (all construction stays
    ahead of run()'s torch.manual_seed), so the sampling RNG stream — and thus
    the output mesh — is unchanged. Set SKIP_INIT_ON_LOAD=0 to restore the
    original init-then-load behavior.
    """
    path = os.path.join(TRELLIS_ROOT, "trellis2/models/__init__.py")
    src = read_file(path)

    if "SKIP_INIT_ON_LOAD" in src:
        print(f"  Already patched: {os.path.relpath(path, TRELLIS_ROOT)}")
        return

    src = src.replace(
        "    with open(config_file, 'r') as f:\n"
        "        config = json.load(f)\n"
        "    model = __getattr__(config['name'])(**config['args'], **kwargs)\n"
        "    model.load_state_dict(load_file(model_file), strict=False)\n",
        "    with open(config_file, 'r') as f:\n"
        "        config = json.load(f)\n"
        "    _klass = __getattr__(config['name'])\n"
        "    # load_state_dict below overwrites every parameter, so the model's\n"
        "    # initialize_weights() (tens of seconds of xavier/normal RNG for the\n"
        "    # 1.3B flow models) is pure waste. Skip it — verified bit-identical.\n"
        "    _skip = os.environ.get('SKIP_INIT_ON_LOAD', '1') != '0' and hasattr(_klass, 'initialize_weights')\n"
        "    if _skip:\n"
        "        _orig_init = _klass.initialize_weights\n"
        "        _klass.initialize_weights = lambda self: None\n"
        "    try:\n"
        "        model = _klass(**config['args'], **kwargs)\n"
        "    finally:\n"
        "        if _skip:\n"
        "            _klass.initialize_weights = _orig_init\n"
        "    model.load_state_dict(load_file(model_file), strict=False)\n",
    )
    write_file(path, src)
