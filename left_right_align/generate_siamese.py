import argparse

parser = argparse.ArgumentParser(description='Generates the replica part of a siamese net prototxt for Caffe.')
parser.add_argument('--source', default=False, help="The source prototxt file.")
parser.add_argument('--target', default=False, help="The target location for the  prototxt file.")
args = parser.parse_args()

def main():
  with open(args.source, 'r') as source_file:
    source_lines = source_file.readlines()
  with open(args.target, 'w') as target_file:
    in_net = False
    prev_line = ''
    replica_net = []
    for source_line in source_lines:

      # Done with Source Net
      if "# REPLICA" in source_line:
        in_net = False
        target_file.write(source_line)
        for ind, line in enumerate(replica_net):
          if ((ind > 0 and "name" in line and "layer" in replica_net[ind-1]) or "top" in line or "bottom" in line):
            target_file.write(line[:-2] + '_rep"\n')
          else:
            target_file.write(line)
      else:
         # Beginning of Source Net
        if 'name: "conv1"' in source_line:
          replica_net.append(prev_line)
          in_net = True

        # During Source Net
        if in_net:
          replica_net.append(source_line)

        target_file.write(source_line)
        prev_line = source_line

main()
