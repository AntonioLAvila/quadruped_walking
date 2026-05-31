python -c "import mjlab_env; from mjlab.scripts.play import main; main()" \
  Mjlab-Velocity-Flat-Unitree-Go2 \
  --agent trained \
  --checkpoint-file logs/rsl_rl/go2_velocity/<run>/model_5000.pt \
  --num-envs 1
