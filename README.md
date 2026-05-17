# Module Template

This is a template for a module repository. A Module is considered to follow these rules:

- This repo is a ros2 package. [See below](#setup-of-the-ros2-package)
- It mainly contains scripts, meshes, launch files, configs, URDF/SRDF xacro macros, that are then used within an applications.
- This repository will be addedd to applications as a submodule.
- Launch files for testing the functionality of this module are welcome.
- Also documentation, that is inherent to this module can be included in this README and in the doc folder, like specific coordinate frames etc.

You can delete these infos if you dont need them anymore. You can always revisit them in the [Wiki](https://gitlab.lrz.de/groups/iwb-tgmr-ros2/-/wikis/home).

## Setup of the ROS2 Package

First you need to create the ros2 workspace with the source folder. Then you create the ros2 package. Then you clone the newly created repository (from the module-template) into this package. A tutorial can be found here: [Create a new ROS2 pkg as a repo](https://gitlab.lrz.de/groups/iwb-tgmr-ros2/-/wikis/Creating-a-new-Repository#new-repository-with-ros2-package). Then you push the changes to the module repository.

## References

- [TGMR ROS2 Wiki](https://gitlab.lrz.de/groups/iwb-tgmr-ros2/-/wikis/home).
- [ROS2 Jazzy Wiki: Creating a package](https://docs.ros.org/en/foxy/Tutorials/Beginner-Client-Libraries/Creating-Your-First-ROS2-Package.html)
