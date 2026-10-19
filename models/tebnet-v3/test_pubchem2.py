import requests
res = requests.get("https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/8003/JSON/").json()
for sec in res['Record']['Section']:
    if sec['TOCHeading'] == 'Chemical and Physical Properties':
        for subsec in sec['Section']:
            if subsec['TOCHeading'] == 'Experimental Properties':
                for prop in subsec['Section']:
                    print(prop['TOCHeading'])
                    if 'Enthalpy of Formation' in prop['TOCHeading']:
                        print(prop)
