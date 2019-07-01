from setuptools import setup

setup(name="fcsbh",
      version="0.0.1",
      description="PCH Analysis of BH data.",
      url="https://github.com/bankanidhi/fcsbh",
      entry_points={
          "console_scripts": [
              "fcsbh=fcsbh.run:run"
          ]
      },
      install_requires=["numpy>=1.14.5",
                        "scipy>=1.1.0",
                        "matplotlib>=2.2.2",
                        "lmfit>=0.9.11"],
      license="MIT",
      zip_safe=False)
