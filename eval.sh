python -c "import mjlab_env; from mjlab.scripts.play import main; main()" \
  Mjlab-Velocity-Flat-Unitree-Go2 \
  --agent trained \
  --checkpoint-file logs/rsl_rl/go2_velocity/feet-time-25/model_999.pt \
  --num-envs 16
