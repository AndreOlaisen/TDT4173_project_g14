import pathlib
import shutil
import subprocess


def get_build_path(root_path):
    root_path = pathlib.Path(root_path)
    if not pathlib.Path(root_path/"CMakeLists.txt").exists():
        raise ValueError(f"Path {root_path} does not contain a CMakeLists.txt!")
    build_path = pathlib.Path(root_path)/"build"
    return build_path.absolute()


def cmake_build(root_path, clean=False):
    build_path = get_build_path(root_path)
    if clean:
        if build_path.exists():
            shutil.rmtree(build_path)
        build_path.mkdir()
        p = subprocess.run(["cmake", ".."], cwd=build_path,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True)
        print(p.stdout)
        p.check_returncode()
    p = subprocess.run(["cmake", "--build", "."], cwd=build_path,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True)
    print(p.stdout)
    p.check_returncode()
