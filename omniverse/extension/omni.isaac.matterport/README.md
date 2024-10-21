# Usage

For the Matterport extension, a GUI interface is available. To use it, start the simulation:

```bash
cd IsaacLab
./isaaclab.sh -s
```

To enable your extension, follow these steps:

1. **Add the search path of your repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon** (☰), then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to `isaaclab_envs/extensions` and `IsaacLab/source/extensions`
    - Click on the **Hamburger Icon** (☰), then click `Refresh`.

2. **Search and enable your extension**:
    - Type `Matterport` in the search bar, and you should see the Matterport3D extension under the `Third Party` category.
    - Toggle it to enable your extension.

To use it as part of an IsaacLab workflow, please refer to the [ViPlanner Demo](https://github.com/leggedrobotics/viplanner/tree/main/omniverse).

**IMPORTANT**: The current GUI demo can only be used to import meshes. There are issues for displaying the images of the added cameras. 
We are working on a fix. 
