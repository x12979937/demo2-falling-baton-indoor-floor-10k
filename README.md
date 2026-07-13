# demo2 falling baton indoor floor 10k

Generated on remote machine 1 under allowed paths only.

Data layout:
- archives/: tar.gz batches of falling baton episodes, split to stay below GitHub's single-file size limit.
- metadata/archive_index.jsonl: archive ranges, sizes, and checksums.
- metadata/generation_config.json: simulation and rendering settings.
- code/render_formal_threedemo_physx_videos.py: collection script snapshot.
- status.json: latest generation/push status.
- each episode folder: mesh_assets/mesh_assets_manifest.json plus visual_mesh.obj and collision_mesh.obj for every baton.

Each episode contains the simulation dataset, collision events, manifest, logs, eight multiview videos with 2D boxes and pixel-mask overlays, per-view raw renderer RGB PNG frames packed as `*_frames.tar.gz`, and `state_replay_validation.json`. Each archive also contains replay_state_exports/<episode>/replay_state.npz and replay_state_manifest.json exported by the validator for IsaacSim PhysX import verification.
