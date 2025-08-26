import json, os, subprocess, sys, tempfile, uuid, wandb
from pathlib import Path

import click


def make_temp_config(base_path, **kwargs) -> str:
    with open(base_path) as f:
        cfg = json.load(f)

    for k, v in kwargs.items():
        cfg[k] = v

    out_path = os.path.join(
        tempfile.gettempdir(),
        f"cfg_{uuid.uuid4().hex[:8]}.json"
    )
    with open(out_path, "w") as f:
        json.dump(cfg, f, indent=2)
    return out_path

def run_trial():
    # 1) get the sweep params
    run = wandb.init(project="bfrb_kaggle")

    with open(os.environ['SWEEP_CONFIG_PATH']) as f:
        sweep_config = json.load(f)

    kwargs = {x: wandb.config.get(x) for x in sweep_config['parameters'].keys()}
    tmp_cfg = make_temp_config(base_path=os.environ['BASE_CONFIG_PATH'], **kwargs)

    # 2) set env so the child 'bfrb.run' resumes THIS run (no double runs)
    env = os.environ.copy()
    env["WANDB_RESUME"] = "allow"
    env["WANDB_RUN_ID"] = run.id
    env["WANDB_PROJECT"] = run.project
    if getattr(run, "entity", None):
        env["WANDB_ENTITY"] = run.entity
    env["WANDB_DIR"] = run.dir  # keep files/artifacts together

    # Close our handle before spawning the child
    wandb.finish()

    # 3) run your unchanged CLI (it calls wandb.init() itself)
    #    because of the env above, it will RESUME the same run
    subprocess.run(
        [sys.executable, "-m", "bfrb.run", "--config-path", tmp_cfg],
        check=True,
        env=env,
    )

@click.command()
@click.option(
    '--sweep-config-path',
    type=click.Path(path_type=Path, readable=True, dir_okay=True),
    required=True,
)
@click.option(
    '--sweep-count',
    type=int,
    required=True,
)
@click.option(
    '--train-config-path',
    type=click.Path(path_type=Path, readable=True, dir_okay=True),
    required=True,
)
def main(sweep_config_path: Path, sweep_count, train_config_path: Path):
    with open(sweep_config_path) as f:
        sweep_config = json.load(f)

    os.environ['BASE_CONFIG_PATH'] = str(train_config_path)
    os.environ['SWEEP_CONFIG_PATH'] = str(sweep_config_path)

    sweep_id = wandb.sweep(sweep_config, project="bfrb_kaggle")
    wandb.agent(sweep_id, function=run_trial, count=sweep_count)

if __name__ == "__main__":
    main()
