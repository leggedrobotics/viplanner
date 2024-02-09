import torch
import yaml
from omni.isaac.matterport.domains import MatterportRayCasterCamera
from omni.isaac.orbit.sensors.ray_caster import RayCasterCameraCfg
from omni.isaac.orbit.utils.configclass import configclass
from omni.viplanner.viplanner import DATA_DIR

from viplanner.config.viplanner_sem_meta import VIPlannerSemMetaHandler


class VIPlannerMatterportRayCasterCamera(MatterportRayCasterCamera):
    def __init__(self, cfg: object):
        super().__init__(cfg)

    def _color_mapping(self):
        viplanner_sem = VIPlannerSemMetaHandler()
        with open(DATA_DIR + "/mpcat40_to_vip_sem.yml") as file:
            map_mpcat40_to_vip_sem = yaml.safe_load(file)
        color = viplanner_sem.get_colors_for_names(list(map_mpcat40_to_vip_sem.values()))
        self.color = torch.tensor(color, device=self._device, dtype=torch.uint8)


@configclass
class VIPlannerMatterportRayCasterCameraCfg(RayCasterCameraCfg):
    class_type = VIPlannerMatterportRayCasterCamera
