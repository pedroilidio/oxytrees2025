# e.g. sh forward_ssh_port.sh dali
for i in {1..10}; do
    ssh -NL 5001:localhost:5001 "$@"
done