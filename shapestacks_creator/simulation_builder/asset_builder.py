"""
Contains builder classes to create asset catalogs fpr simulation environments.
"""

import os
import glob
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from simulation_builder.mj_elements import MjAsset, MjTexture, MjMesh, MjMaterial


class MjAssetBuilder(object):
    """
    An asset builder for MuJoCo asset catalogs. Can load / create a MuJoCo asset
    list, insert assets and export it as MJCF compatible XML.
    """
    TL_TAG_ASSETS = "assets"

    def __init__(self):
        self._assets = None # top-level node for asset catalog
        self._asset = None


    # asset catalog import / export

    def load_assets(self, assets_xml_file: str):
        """
        Loads an asset list from a MuJoCo XML file.
        """
        self._assets = ET.parse(assets_xml_file).getroot()
        self._asset = self._assets.find('asset')

    def new_assets(self):
        """
        Creates a new empty asset catalog.
        """
        self._assets = ET.Element(self.TL_TAG_ASSETS)
        self._asset = ET.SubElement(self._assets, MjAsset().to_etree_elem())

    def export_assets(self, assets_xml_file: str):
        """
        Exports the current asset catalog to the specified file.
        """
        with open(assets_xml_file, 'w') as f:
            xml_str = BeautifulSoup(ET.tostring(self._assets), 'xml').prettify()
            f.write(xml_str)


    # basic asset insertion

    def add_asset(self, asset):
        """
        Adds an asset to the asset catalog.
        """
        asset_elem = asset.to_etree_elem()
        self._asset.append(asset_elem)


    # batch asset creation

    def convert_textures(self, texdir: str):
        """
        Adds all textures (PNG only!) found under texdir to the asset list.
        Naming convention is 'tex_<dir>_..._<file_name>'.
        """
        tex_files = glob.glob(texdir + '/**/*.png', recursive=True)
        for tex in tex_files:
            tex_path = os.path.relpath(tex, texdir)
            tex_name = 'tex_' + '_'.join(tex_path.split('/')).rstrip('.png')
            tex_asset = MjTexture()
            tex_asset.name = tex_name
            tex_asset.type = '2d'
            tex_asset.file = tex_path
            self.add_asset(tex_asset)

    def convert_meshes(self, meshdir: str):
        """
        Adds all meshes (STL only!) found under meshdir to the asset list.
        Naming convention is 'mesh_<dir>_..._<file_name>'.
        """
        raise NotImplementedError


    # catalog inspection

    def get_texture_names(self):
        """
        Returns a list of all texture names in the asset catalog.
        """
        nodes = self._asset.findall(".//texture")
        return [t.attrib["name"] for t in nodes]

    def get_texture_by_name(self, texname: str) -> MjTexture:
        """
        Returns the MjTexture object for the given name.
        """
        raise NotImplementedError

    def get_mesh_names(self):
        """
        Returns a list of all mesh names in the asset catalog.
        """
        nodes = self._asset.findall(".//mesh")
        return [t.attrib["name"] for t in nodes]

    def get_mesh_by_name(self, texname: str) -> MjMesh:
        """
        Returns the MjMesh object for the given name.
        """
        raise NotImplementedError

    def get_material_names(self):
        """
        Returns a list of all material names in the asset catalog.
        """
        nodes = self._asset.findall(".//material")
        return [t.attrib["name"] for t in nodes]

    def get_material_by_name(self, texname: str) -> MjMaterial:
        """
        Returns the MjMaterial object for the given name.
        """
        raise NotImplementedError
