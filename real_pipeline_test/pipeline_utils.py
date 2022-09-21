def get_galaxy_filename(name, directory, prohibited=None):
    for filename in os.listdir(directory):
        if name in filename and (prohibited is None or all([x not in filename for x in prohibited])):
            return directory + "/" + filename