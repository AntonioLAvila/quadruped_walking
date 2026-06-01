python -c "import mjlab_env; from mjlab.scripts.play import main; main()" \
  Mjlab-Velocity-Flat-Unitree-Go2 \
  --agent trained \
  --checkpoint-file logs/rsl_rl/go2_velocity/2026-05-31_17-03-42/model_499.pt \
  --num-envs 32
