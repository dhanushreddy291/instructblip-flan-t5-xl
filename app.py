import beam

app = beam.App(
    name="instructblip",
    cpu=16,
    memory="32Gi",
    gpu="A10G",
    python_packages=[
        "git+https://github.com/NielsRogge/transformers.git@add_instruct_blip",
        "torch>=1.13.1,<2",
        "Pillow",
        "requests",
    ]
)

app.Trigger.RestAPI(
    inputs={"url": beam.Types.String()},
    outputs={"response": beam.Types.String()},
    handler="run.py:generate_caption",
    keep_warm_seconds=60,
)

app.Mount.SharedVolume(name="instructblip-weights", path="./instructblip-weights")
