# Setting up a New SSD in WSL/Linux

## Prerequisites
- New SSD physically installed in the PC (NVMe is preferred if available).
- (Optional) Note down the **serial number** so you can identify it for sure later.

---

## Step 1. Format the Drive to ext4
`ext4` is the most compatible filesystem for Linux, and WSL2 supports it natively. We assume data processing will always be done in WSL/Linux, so we use `ext4`.  

Other options like NTFS or exFAT are more compatible with Windows, but they have worse performance and reliability in WSL. We will also setup backup to Labshare (NTFS), so Windows access can be done from there.

### 1A. Pre-setup when using WSL: Attach Drive to WSL through Windows PowerShell

> Skip this step if you are using a native Linux PC.

This is needed because WSL cannot see the raw drive directly.


1. Open **PowerShell** in **administrator mode**.
2. List all disks:
   ```powershell
   Get-Disk
   ```
   → Identify your new SSD’s **disk number** (e.g., `2`) based on model, size, or serial number.
3. Attach it to WSL (bare mode):
   ```powershell
   wsl --mount \\.\PHYSICALDRIVE<DISK_NUMBER> --bare
   ```

### 1B. Format in WSL/Linux
1. Open **WSL/Linux terminal** and list disks:
   ```bash
   lsblk -o NAME,TYPE,SIZE,FSTYPE,MOUNTPOINT,SERIAL,VENDOR,MODEL
   ```
   → Note the whole device name (e.g., `/dev/sda` or `/dev/nvme0n1`).

2. Set a shell variable for convenience:
   ```bash
   DISK="/dev/sdX"   # Replace sdX with the actual device (e.g. sda)
   ```

3. Create a GPT partition table + one full-size partition:
   ```bash
   sudo parted -s "$DISK" mklabel gpt mkpart primary ext4 0% 100%
   ```

4. Format the new partition:
   ```bash
   sudo mkfs.ext4 "${DISK}1"
   ```

5. Verify:
   ```bash
   lsblk -f
   ```
   → You should see `${DISK}1` with `ext4` as the FSTYPE.

---

## Step 2 (WSL). Mount the Drive (with systemd)

These steps will set up the drive to be automatically attached and mounted in WSL at login of Windows and boot of WSL.
This is kind of complicated because WSL cannot attach the drive directly, so we use a scheduled task in Windows to do it and also call it from a systemd service in WSL.

For Setting up on a native Linux PC, jump to the section below.

### 2A. Create Scheduled Task in Windows
This ensures the disk is attached to WSL on login.

In **PowerShell (Admin)**, run: 

Replace <DISK_NUMBER> with the disk number you noted earlier (e.g., `2`):
```powershell
$taskName = "WSL Attach Drive (User)"
$diskArg  = "--mount \\.\PHYSICALDRIVE<DISK_NUMBER> --partition 1 --type ext4"

Register-ScheduledTask -TaskName $taskName `
-Action   (New-ScheduledTaskAction -Execute "wsl.exe" -Argument $diskArg) `
-Trigger  (New-ScheduledTaskTrigger -AtLogOn) `
-Principal(New-ScheduledTaskPrincipal -UserId "$env:USERNAME" -RunLevel Highest)
```


### 2B. Find the Partition UUID
In **WSL** terminal, grab the UUID of the drive and set it as a variable for later use:
```bash
UUID=$(sudo blkid "${DISK}1" | awk -F'UUID="' '{print $2}' | awk -F'"' '{print $1}')
```

> This used the `$DISK` variable you set earlier in Step 1B.1-2. if you are now in a new terminal, you may need to set it again following the same steps (No need to format the disk again). You can check whether the `$DISK` variable is set by running `echo $DISK`. If it is empty, set it again.

Verify:
```bash
echo "UUID: $UUID"
```
→ You should see something like `UUID: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`.

If you see an error, run `sudo blkid` to check if the disk is attached correctly.


### 2C. Create Attach Script

Create a script to run the scheduled powershell task from WSL and wait for the drive to be visible in WSL.


```bash
[ -n "$UUID" ] || { echo "UUID is empty; run 2B first"; exit 1; }

sudo tee /usr/local/sbin/wsl-attach-drive.sh >/dev/null <<SH
#!/usr/bin/env bash
set -euo pipefail

UUID="$UUID"
TASK="WSL Attach Drive (User)"

"/mnt/c/Windows/System32/schtasks.exe" /run /tn "\$TASK" >/dev/null 2>&1 || true

for i in {1..32}; do
  if ls /dev/disk/by-uuid 1>/dev/null 2>&1 && \
     ls /dev/disk/by-uuid | grep -qi "^${UUID}\$"; then
    exit 0
  fi
  sleep 0.25
done

echo "Drive with UUID \$UUID not visible yet" >&2
exit 1
SH
sudo chmod +x /usr/local/sbin/wsl-attach-drive.sh

```

This creates a file at:
`/usr/local/sbin/wsl-attach-drive.sh`

You can verify the created script by running:
```bash
cat /usr/local/sbin/wsl-attach-drive.sh
```
It should has the UUID you set earlier.

### 2D. Create Systemd Attach Service

Create a systemd service to run the attach script at boot of WSL by running:

```bash
sudo tee /etc/systemd/system/drive-attach.service >/dev/null <<'UNIT'
[Unit]
Description=Attach drive to WSL via Windows scheduled task
Wants=network-online.target
After=network-online.target

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/wsl-attach-drive.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
UNIT
```

This creates a file at:
`/etc/systemd/system/drive-attach.service`


### 2E. Create Systemd Mount Service

First, define your mount directory:
```bash
MOUNTDIR="/mnt/data"
```
Change this if you want a different mount path.
> Make sure there is no existing file in this path

Then create a native systemd mount unit that mounts the new drive to your desired location by running:

```bash
sudo mkdir -p "$MOUNTDIR"
sudo tee "/etc/systemd/system/$(echo $MOUNTDIR | sed 's|/|-|g' | sed 's|^-||').mount" >/dev/null <<UNIT
[Unit]
Description=Mount SSD at $MOUNTDIR
Requires=drive-attach.service
After=drive-attach.service

[Mount]
What=/dev/disk/by-uuid/$UUID
Where=$MOUNTDIR
Type=ext4
Options=defaults

[Install]
WantedBy=multi-user.target
UNIT
```

This creates a file at:
`/etc/systemd/system/$(echo $MOUNTDIR | sed 's|/|-|g' | sed 's|^-||').mount`

You can verify the created mount unit by running:
```bash
systemctl cat "$(echo $MOUNTDIR | sed 's|/|-|g' | sed 's|^-||').mount"
```
It should use the UUID and mount path you set earlier.

### 2F. Enable and Start
```bash
sudo systemctl daemon-reload
sudo systemctl enable drive-attach.service "$(echo $MOUNTDIR | sed 's|/|-|g' | sed 's|^-||').mount"
sudo systemctl start drive-attach.service "$(echo $MOUNTDIR | sed 's|/|-|g' | sed 's|^-||').mount"
```

Verify:
```bash
systemctl status drive-attach.service --no-pager
systemctl status "$(echo $MOUNTDIR | sed 's|/|-|g' | sed 's|^-||').mount" --no-pager
df -h "$MOUNTDIR"
```

---

## Step 2 (Native Linux). Mount the Drive
This is much simpler than WSL, as it can see the drive directly.
### 2A. Find the UUID of the drive
```bash
lsblk -f
```
Copy the UUID of the data drive

### 2B. Make a director for mounting
```bash
sudo mkdir -p /mnt/Data
```
You can replace `Data` with your prefered folder name.

### 2C. Add the mount to `fstab`
Edit the fstab file
```bash
sudo nano /etc/fstab
```

Add a line at the bottom (replace UUID and mount point accordingly):
```
UUID=<your_UUID> /mnt/Data ext4 defaults 0 2
```
Save and exit (`Ctrl+O`, `Enter`, `Ctrl+X`).

### 2D. Test your `fstab` entry and change permissions
```bash
sudo mount -a
```
If no errors appear, you’re good to go — it’ll mount automatically on boot.

Change to writer permission

```bash
sudo chown -R $USER:$USER /mnt/Data
```



---

## Step 3 (WSL). Set up Automated Backup to Labshare
(Instructions TBD.)

---

## Step 3 (Native Linux). Set up Automated Backup to Labshare
(Instructions TBD.)

---

## Step 4. (Optional) Setup NFS Share

This allows you to share the new drive over the local network to other PCs running WSL/Linux, so you can process data 
on a separate computer or running experiments there while still saving data to the same physical drive.

### 4A. Pre-setup when using WSL: Configure WSL to Use Bridged Networking
> Skip this step if you are using a native Linux PC.

By default, WSL uses a NAT network, which means it cannot be directly accessed by other computers on the network.
We need to set up WSL to use a bridged network connection, so it is exposed to the network that the windows host is connected to as if it is a new computer in the network.

1. Enable required Windows features.

   In **PowerShell (Admin)**, run:
   ```powershell
   Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V -All
   Enable-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform
   Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux
   Enable-WindowsOptionalFeature -Online -FeatureName HypervisorPlatform
   ```
   → Each step may require a reboot, you can choose `N` for the first three steps and reboot only once at the end.

2. Create a new Hyper-V switch (still in **PowerShell (Admin)**):

   First, find the name of the network adapter you want to bridge to (e.g., `Ethernet`, `Ethernet 2`, `Wi-Fi`):
   ```powershell
   Get-NetAdapter | Where-Object {$_.Status -eq "Up"} | Select Name,InterfaceDescription
   ```

   Then create the switch :
   ```powershell
   
   New-VMSwitch -Name "WSLBridge" -NetAdapterName "<NIC_NAME>" -AllowManagementOS $true
   ```
   Replace `<NIC_NAME>` with the name of your active network adapter (e.g., `Ethernet`, `Wi-Fi`).
   > ⚠ This will briefly disconnect the network.

3. Set up WSL to use the new switch and generate a unique MAC address:
   
   In **PowerShell (Admin)**, run:
   ```powershell
   # Generate a unique MAC using Hyper-V OUI (00-15-5D) + random tail
    $rand = Get-Random -Maximum 0xFFFFFF
    $mac  = "00-15-5D-{0:X2}-{1:X2}-{2:X2}" -f (($rand -shr 16) -band 0xFF), (($rand -shr 8) -band 0xFF), ($rand -band 0xFF)
    
    # Path to .wslconfig
    $wslconfig = "$env:USERPROFILE\.wslconfig"
    
    # Write config (overwrite if exists)
    @"
    [wsl2]
    networkingMode=bridged
    vmSwitch=WSLBridge
    macAddress=$mac
    "@ | Set-Content -Encoding ASCII $wslconfig
    
    Write-Host "Generated MAC: $mac"
    Write-Host "Saved to $wslconfig"
   ```
   > The random MAC address generation is kind of a hack to ensure WSL gets a unique MAC address, so that the WSL instance can get a fixed and unique IP address from the DHCP server of the bridged network.
   
   <sup>God this whole WSL setup is becoming more and more of a headache. This works if you can bear through it (we are near the end), but we should really consider fully switching to a native Linux PC...</sup>
   

4. Restart WSL (still in PowerShell):
   ```powershell
   wsl --shutdown
   ```

5. Verify in **WSL terminal**:
   ```bash
   ip addr
   ```
   → You should see an IP on `eth0` that is in the same range as your host network (e.g., `10.66.xx.xx`).

   We will use this IP address on the client PC to mount the NFS share, so note it down.

### 4B. Set up NFS Server (on host WSL or Linux PC)

In **WSL** (or native  **Linux**) terminal, set up the NFS server to share the mounted drive.

1. Install NFS server:
   ```bash
   sudo apt install nfs-kernel-server
   ```
2. Export the local mount directory:
   ```bash
   sudo nano /etc/exports
   ```
   add
   ```bash
   <MOUNTDIR> \
     <client_ip_1>(rw,sync,no_subtree_check,all_squash,anonuid=1000,anongid=1000) \
     <client_ip_2>(rw,sync,no_subtree_check,all_squash,anonuid=1000,anongid=1000)
   ```

   Replace `<MOUNTDIR>` with the actual mount path where your new drive is mounted (e.g., `/mnt/data`).
   
   Replace `<client_ip_1>` and `<client_ip_2>` with the IP addresses of the client machines that should have access.
   
   > The option `rw` allows read-write access, if you want read-only access for safety, use `ro` instead.
   
   > If the client is WSL, make sure to do step 4A on the client computer, and use the **IP address shown inside WSL**, 
   > not the Windows host IP.

3. Restart NFS:
   ```bash
   sudo systemctl restart nfs-kernel-server
   ```

### 4C. Mount from Client Linux PC (on clinet WSL or Linux PC)

In the **remote** WSL or Linux terminal:
> On WSL, step 4A must be performed first on the client PC as well. 

1. Install NFS tools on the client:
   ```bash
   sudo apt install nfs-common
   ```


2. Define the  the IP address of the host,  the mount directory on the host and local directory where you want to mount the NFS share:
   ```bash
   # IP of the machine exporting NFS (your WSL/host with the data)
   SERVER_IP="10.66.xx.xx"          # ← change me

   # Exported directory on the server (the folder where the new drive is mounted to on the host)
   SERVER_MOUNT_DIR="/mnt/data"          # ← change me if different

   # Where to mount it on this client
   LOCAL_MOUNT_DIR="/mnt/data_from_yyy"      # ← change me to your prefered name
   ```
   > If the remote folder in on wsl, make sure to use the __ip of the wsl, not the windows system__

   > Make sure there is no existing file in `LOCAL_MOUNT_DIR`
   

3. Mount it now:
   ```bash
   sudo mkdir -p "$LOCAL_MOUNT_DIR"
   sudo mount -t nfs -o noatime,actimeo=1 "$SERVER_IP:$SERVER_MOUNT_DIR" "$LOCAL_MOUNT_DIR"
   ```
   The `noatime` removes the access time updates, while `actimeo=1` sets the attribute cache timeout to 1 second. These settings are helpful when viewing the data on the client PC with `acadia_gui`, as they remove the overhead of updating access times and ensure the client sees the latest data quickly.

   Verify:
   ```bash
   df -h "$LOCAL_MOUNT_DIR"
   ```


4. Make it persistent (`/etc/fstab` on the client):
   ```bash
   echo "$SERVER_IP:$SERVER_MOUNT_DIR $LOCAL_MOUNT_DIR nfs noatime,actimeo=1 0 0" | sudo tee -a /etc/fstab
   ```



### Unmounting the NFS Share

   - To unmount the NFS share on the client, run:

      ```bash
      sudo umount "$LOCAL_MOUNT_DIR"
      ```

   - remove the entry from `/etc/fstab` if you want to stop auto-mounting:

      ```bash
      sudo sed -i "\|$SERVER_IP:$SERVER_MOUNT_DIR|d" /etc/fstab
      ```


---
