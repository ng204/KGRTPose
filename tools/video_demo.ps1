# Define the base names of the videos to process
$videoBases = @(
    "twoperson-right",
    "twoperson-left",
    "oneperson-1-left",
    "oneperson-2-left",
    "oneperson-1-right",
    "oneperson-2-right"
)

# Define the suffixes for the video files
$suffixes = @("128x128-64x64-256-20241128", "128x128-32x32-256-20241128", "128x128-16x16-256-20241128")

# Define the paths to the models that will be used for processing
$modelPaths = @(
    ".\runs\spipose\train\twoperson\spipose-n-128x128-64x64-256-20241128-20241214\weights\best.pt",
    ".\runs\spipose\train\twoperson\spipose-n-128x128-32x32-256-20241128-20241214\weights\best.pt",
    ".\runs\spipose\train\twoperson\spipose-n-128x128-16x16-256-20241128-20241214\weights\best.pt"
)

# Loop through each video base name and its corresponding suffix
foreach ($base in $videoBases) {
    for ($i = 0; $i -lt $suffixes.Length; $i++) {
        # Construct the full path to the video file
        $videoPath = Join-Path ".\runs\videos\spi_videos" "$base-$($suffixes[$i]).avi"

        # Check if the video file exists before attempting to process it
        if (Test-Path $videoPath) {
            # Construct the command to run pose_video_demo.py with the appropriate parameters
            $command = "python .\tools\pose_video_demo.py --source `"$videoPath`" --model `"$($modelPaths[$i])`" --imgsz 640 --dataset twoperson --kpt-line"

            # Execute the constructed command
            Invoke-Expression $command

            Write-Output "Processed: $videoPath using model: $($modelPaths[$i])"
        } else {
            Write-Warning "Video file not found: $videoPath"
        }
    }
}