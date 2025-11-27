from skbuild import setup


setup(
    packages=["sdu_estimators", "sdu_estimators-stubs"],
    package_dir={"": "python"},
    zip_safe=False,
    cmake_args=[
        "-DBUILD_TESTING=OFF",
        "-DBUILD_DOCS=OFF",
    ],
    cmake_install_dir="python/sdu_estimators",
    include_package_data=True,
    package_data={"sdu_estimators-stubs": ["*.pyi"]}
)

