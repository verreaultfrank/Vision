using System.Collections;
using UnityEngine;
using UnityEditor;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.Features2dModule;
using OpenCVForUnity.Calib3dModule;
using OpenCVForUnity.UnityUtils;
using System.Collections.Generic;
using System.Linq;
using System;
using System.IO;
public class StereoscopicReconstruction : MonoBehaviour
{
    public float baseline = 0.2f;
    public float focalLength = 6f;
    public float radius = 2f;
    public int steps = 4;
    public float maxDepth = 1.8f;
    public float minDepth = 1f;
    private Camera camera1;
    private Camera camera2;
    private Vector3[] points;

    public string pathPicture1;
    public string pathPicture2; 

    public float focalLengthCamera1;
    public float focalLengthCamera2; 

    public float sensorSizeCamera1X;
    public float sensorSizeCamera2X;

    
    public float sensorSizeCamera1Y;
    public float sensorSizeCamera2Y;

    public bool depthFromFeatures = false;

    public bool fromRealImages = false;
    void PositionCameras(float angle)
    {
        // Calculate the point on the circle
        Vector3 circlePoint = new Vector3(Mathf.Cos(Mathf.Deg2Rad * angle), 0, Mathf.Sin(Mathf.Deg2Rad * angle)) * radius;

        //Debug.Log($"Circle point: {circlePoint}");

        // Create a transform to represent the circle point
        GameObject temp = new GameObject("TempPoint");
        temp.transform.position = circlePoint;

        // Make the point look at the origin
        temp.transform.LookAt(Vector3.zero);

        // Calculate camera positions based on baseline
        Vector3 camera1Position = circlePoint - temp.transform.right * (baseline / 2f);
        Vector3 camera2Position = circlePoint + temp.transform.right * (baseline / 2f);

        // Set the positions and rotations of the cameras
        camera1.transform.position = camera1Position;
        camera2.transform.position = camera2Position;
        camera1.transform.rotation = temp.transform.rotation;
        camera2.transform.rotation = temp.transform.rotation;
    

        // Clean up the temporary transform
        Destroy(temp);
    }

    Vector3[] AddPoint(Vector3[] points, Vector3 point)
    {
        // Resize the array
        Vector3[] newPoints = new Vector3[points.Length + 1];

        // Copy the old points
        for (int i = 0; i < points.Length; i++)
        {
            newPoints[i] = points[i];
        }

        // Add the new point
        newPoints[points.Length] = point;

        return newPoints;
    }

    Vector3[] CombinePoints(Vector3[] points1, Vector3[] points2)
    {
        // Resize the array
        Vector3[] newPoints = new Vector3[points1.Length + points2.Length];

        // Copy the old points
        for (int i = 0; i < points1.Length; i++)
        {
            newPoints[i] = points1[i];
        }

        // Add the new points
        for (int i = 0; i < points2.Length; i++)
        {
            newPoints[points1.Length + i] = points2[i];
        }

        return newPoints;
    }

    Texture2D CaptureCameraImage(Camera camera)
    {
        // Create a temporary RenderTexture
        RenderTexture tempRT = new RenderTexture(256, 256, 24);
        camera.targetTexture = tempRT;
        camera.Render();

        // Read the image from the RenderTexture
        RenderTexture.active = tempRT;
        Texture2D image = new Texture2D(256, 256, TextureFormat.RGB24, false);
        image.ReadPixels(new UnityEngine.Rect(0, 0, 256, 256), 0, 0);
        image.Apply();
        RenderTexture.active = null;

        // Clean up
        camera.targetTexture = null;
        Destroy(tempRT);

        return image;
    }
    //Cant pass the disparity value in the texture since it is normalized by unity
    public struct DisparityResult
    {
        public Texture2D disparityMap;
        public float minDisparity;
        public float maxDisparity;
    }

    DisparityResult CalculateDisparity(Texture2D leftImage, Texture2D rightImage)
    {
        Texture2D disparityMap = new Texture2D(leftImage.width, leftImage.height, TextureFormat.RGB24, false);
        disparityMap.SetPixels(new Color[disparityMap.width * disparityMap.height]); 
        
        // Convert Texture2D to OpenCV Mat
        Mat leftMat = new Mat(leftImage.height, leftImage.width, CvType.CV_8UC3);
        Mat rightMat = new Mat(rightImage.height, rightImage.width, CvType.CV_8UC3);
        Utils.texture2DToMat(leftImage, leftMat);
        Utils.texture2DToMat(rightImage, rightMat);

        // Detect ORB features
        ORB orb = ORB.create();
        MatOfKeyPoint keypoints1 = new MatOfKeyPoint();
        MatOfKeyPoint keypoints2 = new MatOfKeyPoint();
        Mat descriptors1 = new Mat();
        Mat descriptors2 = new Mat();
        orb.detectAndCompute(leftMat, new Mat(), keypoints1, descriptors1);
        orb.detectAndCompute(rightMat, new Mat(), keypoints2, descriptors2);

        // Match features
        BFMatcher matcher = BFMatcher.create(Core.NORM_HAMMING);
        MatOfDMatch matches = new MatOfDMatch();
        matcher.match(descriptors1, descriptors2, matches);

        // Find good matches and calculate disparity range
        float maxDisp = float.MinValue;
        float minDisp = float.MaxValue;
        List<DMatch> goodMatches = new List<DMatch>();
        
        for (int i = 0; i < matches.rows(); i++)
        {
            if (matches.toList()[i].distance < 30)
            {
                Point pt1 = keypoints1.toList()[matches.toList()[i].queryIdx].pt;
                Point pt2 = keypoints2.toList()[matches.toList()[i].trainIdx].pt;
                float disparity = (float)(pt1.x - pt2.x);
                
                // Only consider positive disparities (points in front of cameras)
                if (disparity > 0)
                {
                    goodMatches.Add(matches.toList()[i]);
                    maxDisp = Mathf.Max(maxDisp, disparity);
                    minDisp = Mathf.Min(minDisp, disparity);
                }
            }
        }

        //Debug.Log($"Disparity range - Min: {minDisp}, Max: {maxDisp}");

        // Calculate disparity map with normalized values
        for (int i = 0; i < goodMatches.Count; i++)
        {
            Point pt1 = keypoints1.toList()[goodMatches[i].queryIdx].pt;
            Point pt2 = keypoints2.toList()[goodMatches[i].trainIdx].pt;
            float disparity = (float)(pt1.x - pt2.x);
            
            if (disparity > 0)
            {
                // Store the raw disparity value
                float normalizedDisparity = (disparity - minDisp) / (maxDisp - minDisp);
                disparityMap.SetPixel((int)pt1.x, (int)pt1.y, new Color(normalizedDisparity, 0, 0));
          //      Debug.Log($"Match {i}: Point1({pt1.x}, {pt1.y}), Point2({pt2.x}, {pt2.y}), Disparity: {disparity}");
            }
        }

        disparityMap.Apply();
        DisplayImages(leftImage, rightImage, disparityMap);
        return new DisparityResult 
        { 
            disparityMap = disparityMap,
            minDisparity = minDisp,
            maxDisparity = maxDisp
        };
    }

    public DisparityResult CalculateDisparityWithoutFeatures(Texture2D leftImage, Texture2D rightImage)
    {
        Texture2D disparityMap = new Texture2D(leftImage.width, leftImage.height, TextureFormat.RGB24, false);
        disparityMap.SetPixels(new Color[disparityMap.width * disparityMap.height]);

        // Convert Texture2D to OpenCV Mat
        Mat leftMat = new Mat(leftImage.height, leftImage.width, CvType.CV_8UC3);
        Mat rightMat = new Mat(rightImage.height, rightImage.width, CvType.CV_8UC3);
        Utils.texture2DToMat(leftImage, leftMat);
        Utils.texture2DToMat(rightImage, rightMat);

        // Convert to grayscale
        Mat leftGray = new Mat();
        Mat rightGray = new Mat();
        Imgproc.cvtColor(leftMat, leftGray, Imgproc.COLOR_RGB2GRAY);
        Imgproc.cvtColor(rightMat, rightGray, Imgproc.COLOR_RGB2GRAY);

        // Parameters for StereoBM
        int numDisparities = 16 * 4; // Must be divisible by 16
        int blockSize = 7;
        
        // Create StereoBM object
        StereoBM stereo = StereoBM.create(numDisparities, blockSize);
        
        // Optional parameters to improve quality
        stereo.setPreFilterSize(5);
        stereo.setPreFilterCap(61);
        stereo.setMinDisparity(0);
        stereo.setTextureThreshold(507);
        stereo.setUniquenessRatio(0);
        stereo.setSpeckleWindowSize(0);
        stereo.setSpeckleRange(8);
        stereo.setDisp12MaxDiff(1);

        // Calculate disparity
        Mat disparity = new Mat();
        stereo.compute(leftGray, rightGray, disparity);

        // Convert disparity to float for better precision
        Mat disparityFloat = new Mat();
        disparity.convertTo(disparityFloat, CvType.CV_32F, 1.0/16.0); // Scale factor of 16 because OpenCV stores disparity values multiplied by 16

        // Find min and max disparity values
        Core.MinMaxLocResult minMax = Core.minMaxLoc(disparityFloat);
        float minDisparity = (float)minMax.minVal;
        float maxDisparity = (float)minMax.maxVal;

        //Debug.Log($"Raw disparity range - Min: {minDisparity}, Max: {maxDisparity}");

        // Convert disparity to texture
        for (int y = 0; y < disparityFloat.rows(); y++)
        {
            for (int x = 0; x < disparityFloat.cols(); x++)
            {
                float disparityValue = (float)disparityFloat.get(y, x)[0];
                if (disparityValue > 0) // Only consider valid disparity values
                {
                    // Normalize disparity value to [0,1] range
                    float normalizedDisparity = (disparityValue - minDisparity) / (maxDisparity - minDisparity);
                    disparityMap.SetPixel(x, y, new Color(normalizedDisparity, 0, 0));
                }
            }
        }

        disparityMap.Apply();
        DisplayImages(leftImage, rightImage, disparityMap);

        return new DisparityResult
        {
            disparityMap = disparityMap,
            minDisparity = minDisparity,
            maxDisparity = maxDisparity
        };
    }

    Vector3[] CalculateDepthMap()
    {
        // Capture images from both cameras
        Texture2D leftImage = CaptureCameraImage(camera1);
        Texture2D rightImage = CaptureCameraImage(camera2);

        DisparityResult disparityResult;
        // Calculate disparity map
        if(depthFromFeatures)
            disparityResult = CalculateDisparity(leftImage, rightImage);
        else
            disparityResult = CalculateDisparityWithoutFeatures(leftImage, rightImage);
        
        Vector3[] newPoints = new Vector3[0];
            
        // Calculate pixels per millimeter on the sensor
        float pixelsPerMillimeterX = leftImage.width / camera1.sensorSize.x;
        float pixelsPerMillimeterY = leftImage.height / camera1.sensorSize.y;        
        float focalLengthPixels = camera1.focalLength * pixelsPerMillimeterX;

        Debug.Log($"Camera Parameters:");
        Debug.Log($"  Position: {camera1.transform.position}");
        Debug.Log($"  Distance from origin: {camera1.transform.position.magnitude}");
        Debug.Log($"  Focal Length (mm): {camera1.focalLength}");
        Debug.Log($"  Pixels per mm x: {pixelsPerMillimeterX}");
        Debug.Log($"  Pixels per mm y: {pixelsPerMillimeterY}");
        Debug.Log($"  Baseline (world units): {baseline}");
        Debug.Log($"  Max Depth: {maxDepth}");

        int pointsOverMaxDepth = 0;
        float minCalculatedDepth = float.MaxValue;
        float maxCalculatedDepth = float.MinValue;

        for (int y = 0; y < disparityResult.disparityMap.height; y++)
        {
            for (int x = 0; x < disparityResult.disparityMap.width; x++)
            {
                Color pixel = disparityResult.disparityMap.GetPixel(x, y);
                
                if (pixel.r > 0)  // Only process pixels where we have disparity
                {
                    float disparityPixels = pixel.r * (disparityResult.maxDisparity - disparityResult.minDisparity) + disparityResult.minDisparity;
                    
                    // Calculate depth using focal length in pixels (same as we use for depth)
                    float depth = (baseline * focalLengthPixels) / disparityPixels;

                    // Discard points beyond the maximum depth
                    if (depth <= maxDepth && depth >= minDepth)
                    {
                        minCalculatedDepth = Mathf.Min(minCalculatedDepth, depth);
                        maxCalculatedDepth = Mathf.Max(maxCalculatedDepth, depth);

                        // Keep everything in pixels for ray direction
                        float pixelX = x - disparityResult.disparityMap.width/2f;
                        float pixelY = disparityResult.disparityMap.height/2f - y;

                        // Create ray direction in camera space
                        Vector3 rayDir = new Vector3(
                            pixelX / focalLengthPixels,
                            pixelY / focalLengthPixels,
                            1.0f  // Forward in camera space is +Z
                        ).normalized;

                        // Camera space point
                        Vector3 cameraSpacePoint = rayDir * depth;

                        // Transform to world space
                        Vector3 worldPosition = camera1.transform.TransformPoint(cameraSpacePoint);

                        newPoints = AddPoint(newPoints, worldPosition);
                    }
                    
                }
            }
        }
        
        Debug.Log($"Depth Statistics:");
        Debug.Log($"  Points over maxDepth: {pointsOverMaxDepth}");
        Debug.Log($"  Min calculated depth: {minCalculatedDepth}");
        Debug.Log($"  Max calculated depth: {maxCalculatedDepth}");
        Debug.Log($"  Total valid points: {newPoints.Length}");

        return newPoints;
    }

    Vector3[] CalculateDepthMapFromRealPictures()
    {
        // Load images from file paths
        Texture2D leftImage = LoadImageFromFile(pathPicture1);
        Texture2D rightImage = LoadImageFromFile(pathPicture2);

        // Calculate disparity map
        DisparityResult disparityResult;
        if(depthFromFeatures)
            disparityResult = CalculateDisparity(leftImage, rightImage);
        else
            disparityResult = CalculateDisparityWithoutFeatures(leftImage, rightImage);

        Vector3[] newPoints = new Vector3[0];

        // Calculate pixels per millimeter on the sensor
        float pixelsPerMillimeterX1 = leftImage.width / sensorSizeCamera1X;
        float pixelsPerMillimeterY1 = leftImage.height / sensorSizeCamera1Y;
        float pixelsPerMillimeterX2 = rightImage.width / sensorSizeCamera2X;
        float pixelsPerMillimeterY2 = rightImage.height / sensorSizeCamera2Y;

        // Calculate focal lengths in pixels
        float focalLengthPixels1 = focalLengthCamera1 * pixelsPerMillimeterX1;
        float focalLengthPixels2 = focalLengthCamera2 * pixelsPerMillimeterX2;

        float minCalculatedDepth = float.MaxValue;
        float maxCalculatedDepth = float.MinValue;

        for (int y = 0; y < disparityResult.disparityMap.height; y++)
        {
            for (int x = 0; x < disparityResult.disparityMap.width; x++)
            {
                Color pixel = disparityResult.disparityMap.GetPixel(x, y);

                if (pixel.r > 0)  // Only process pixels where we have disparity
                {
                    float disparityPixels = pixel.r * (disparityResult.maxDisparity - disparityResult.minDisparity) + disparityResult.minDisparity;

                    // Calculate depth using average focal length in pixels
                    float averageFocalLengthPixels = (focalLengthPixels1 + focalLengthPixels2) / 2f;
                    float depth = (baseline * averageFocalLengthPixels) / disparityPixels;

                    // Discard points beyond the maximum depth (and below minimum depth)
                    if (depth <= maxDepth && depth >= minDepth) 
                    {
                        minCalculatedDepth = Mathf.Min(minCalculatedDepth, depth);
                        maxCalculatedDepth = Mathf.Max(maxCalculatedDepth, depth);

                        // Keep everything in pixels for ray direction
                        float pixelX = x - disparityResult.disparityMap.width/2f;
                        float pixelY = disparityResult.disparityMap.height/2f - y;

                        // Create ray direction in camera space (assuming camera1 is the left camera)
                        Vector3 rayDir = new Vector3(
                            pixelX / focalLengthPixels1, 
                            pixelY / focalLengthPixels1, 
                            1.0f  // Forward in camera space is +Z
                        ).normalized;

                        // Camera space point
                        Vector3 cameraSpacePoint = rayDir * depth;

                        // Transform to world space (assuming camera1 is the left camera)
                        Vector3 worldPosition = camera1.transform.TransformPoint(cameraSpacePoint);

                        newPoints = AddPoint(newPoints, worldPosition);
                    }
                }
            }
        }

        Debug.Log($"Depth Statistics:");
        Debug.Log($"  Min calculated depth: {minCalculatedDepth}");
        Debug.Log($"  Max calculated depth: {maxCalculatedDepth}");
        Debug.Log($"  Total valid points: {newPoints.Length}");

        return newPoints;
    }

    // Helper function to load images from file paths
    private Texture2D LoadImageFromFile(string path)
    {
        Texture2D texture = new Texture2D(2, 2); // Create an empty texture
        byte[] fileData = File.ReadAllBytes(path);
        texture.LoadImage(fileData);
        texture.Apply();
        return texture;
    }

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        //Initialization
        points = new Vector3[0]; // Initialize as an empty array

        // 1. Create two cameras with same focals and separated by an horizontal distance.
        camera1 = new GameObject("Camera1").AddComponent<Camera>();
        camera2 = new GameObject("Camera2").AddComponent<Camera>();
        camera1.focalLength = focalLength;
        camera2.focalLength = focalLength;
        camera1.usePhysicalProperties = true;  // Add this line
        camera2.usePhysicalProperties = true;  // Add this line
        // Set sensor size - let's try a smaller sensor size to get a wider view
        camera1.sensorSize = new Vector2(12, 12); // smaller sensor = wider view
        camera2.sensorSize = new Vector2(12, 12);

        // Set the aperture to ensure enough light
        camera1.aperture = 2.8f;
        camera2.aperture = 2.8f;

        // Ensure the near and far clip planes are reasonable
        camera1.nearClipPlane = 0.01f;
        camera2.nearClipPlane = 0.01f;
        camera1.farClipPlane = 10f;
        camera2.farClipPlane = 10f;

        // Position the cameras a few units from the object
        PositionCameras(0);

        StartCoroutine(Reconstruct());
    }

    void DisplayImages(Texture2D leftImage, Texture2D rightImage, Texture2D disparityMap)
    {
        // Create a new editor window
        ImageDisplayWindow window = EditorWindow.GetWindow<ImageDisplayWindow>("Image Display");
        window.leftImage = leftImage;
        window.rightImage = rightImage;
        window.disparityMap = disparityMap;
    }

    public GameObject CreateGameObjectWithMesh(Mesh mesh, string name = "Generated Mesh", Material material = null)
    {
        // Create new GameObject
        GameObject meshObject = new GameObject(name);
        
        // Add required components
        MeshFilter meshFilter = meshObject.AddComponent<MeshFilter>();
        MeshRenderer meshRenderer = meshObject.AddComponent<MeshRenderer>();
        
        // Generate and assign the mesh
        meshFilter.mesh = mesh;
        
        // Assign material
        if (material == null)
        {
            // Create a default material if none is provided
            material = Resources.Load<Material>("Brick4_Diffuse");
        }
        meshRenderer.material = material;

        return meshObject;
    }
    Vector3[] allPointsCombined = new Vector3[0];
    
    private Vector3[] DownsamplePoints(Vector3[] inputPoints, int targetCount)
    {
        if (inputPoints.Length <= targetCount) return inputPoints;
        int skip = Mathf.Max(1, inputPoints.Length / targetCount);
        return inputPoints.Where((_, index) => index % skip == 0).ToArray();
    }


    IEnumerator Reconstruct()
    {
        // 3. Translate the two cameras around the object.
        for (int i = 0; i < steps; i++)
        {
            float angle = 360f * i / steps;
            PositionCameras(angle);
            Vector3[] newPoints;
            // 4. Calculate the depth map of the scene.
            if(fromRealImages){
                newPoints = CalculateDepthMapFromRealPictures();
                i = steps;
                //We only test with one frame.
            }
            else
                newPoints = CalculateDepthMap();
            
            // Combine points into a single array
            allPointsCombined = CombinePoints(allPointsCombined, newPoints);       


            // Yield control to allow the frame to render
            yield return null; 
        }

        // Initialize min and max values
        Vector3 min = allPointsCombined[0];
        Vector3 max = allPointsCombined[0];

        // Find min and max for x, y, and z
        foreach (Vector3 point in allPointsCombined)
        {
            min.x = Mathf.Min(min.x, point.x);
            min.y = Mathf.Min(min.y, point.y);
            min.z = Mathf.Min(min.z, point.z);

            max.x = Mathf.Max(max.x, point.x);
            max.y = Mathf.Max(max.y, point.y);
            max.z = Mathf.Max(max.z, point.z);
        }
        
        // Use CombinePoints method from earlier in the class
        GameObject mainObject = GameObject.Find("Main"); // Find the GameObject by name
        DelaunayMeshGenerator generator = mainObject.AddComponent<DelaunayMeshGenerator>();//Doing this to enable Loging
        allPointsCombined = DownsamplePoints(allPointsCombined, 4000);
        CreateGameObjectWithMesh(generator.GenerateMesh(allPointsCombined));
    }
}

// Separate class for the editor window
public class ImageDisplayWindow : EditorWindow
{
    public Texture2D leftImage;
    public Texture2D rightImage;
    public Texture2D disparityMap;

    private Vector2 scrollPosition;

    void OnGUI()
    {
        scrollPosition = GUILayout.BeginScrollView(scrollPosition);

        if (leftImage != null)
        {
            GUILayout.Label("Left Image");
            GUILayout.Box(leftImage);
        }

        if (rightImage != null)
        {
            GUILayout.Label("Right Image");
            GUILayout.Box(rightImage);
        }

        // Create a new Texture2D with the same dimensions as the original
        Texture2D flippedDisparityMap = new Texture2D(disparityMap.width, disparityMap.height);

        // Loop through each pixel and flip its X coordinate
        for (int y = 0; y < disparityMap.height; y++)
        {
            for (int x = 0; x < disparityMap.width; x++)
            {
                int flippedY = disparityMap.height - y - 1;
                Color originalColor = disparityMap.GetPixel(x, y);
                flippedDisparityMap.SetPixel(x, flippedY, originalColor);
            }
        }
        flippedDisparityMap.Apply();

        if (flippedDisparityMap != null)
        {
            GUILayout.Label("Disparity Map");
            GUILayout.Box(flippedDisparityMap);
        }

        GUILayout.EndScrollView();
    }
}

public class DelaunayMeshGenerator : MonoBehaviour
{
    private Vector3[] points;

    public DelaunayMeshGenerator()
    {
    }

    void OnDrawGizmos()
    {
        if (points != null)
        {
            // Draw original points in red
            Gizmos.color = Color.red;
            foreach (Vector3 point in points)
            {
                Gizmos.DrawSphere(point, 0.01f);
            }

            //// If we have a generated mesh, draw its vertices in blue
            //if (generatedMesh != null && generatedMesh.vertices.Length > 0)
            //{
                //Gizmos.color = Color.blue;
                //foreach (Vector3 vertex in generatedMesh.vertices)
                //{
                    //// Transform the vertex by the object's transform
                    //Vector3 worldPos = transform.TransformPoint(vertex);
                    //Gizmos.DrawSphere(worldPos, 0.02f);
                //}
            //}
        }
    }

    public Mesh GenerateMesh(Vector3[] points)
    {
        this.points = points;
        if (points.Length < 4)
        {
            Debug.LogError("At least 4 points are required to generate a 3D mesh.");
            return null;
        }

        DataStructures.ViliWonka.KDTree.KDTree kdTree = new DataStructures.ViliWonka.KDTree.KDTree(points);
    
        // Example usage: Find nearest neighbor to a given point
        Vector3 testPoint = new Vector3(0.5f, 0.5f, 0.5f); // Example test point
        DataStructures.ViliWonka.KDTree.KDQuery query = new DataStructures.ViliWonka.KDTree.KDQuery();
        List<int> results = new List<int>();
        query.ClosestPoint(kdTree, testPoint, results);

        List<Tetrahedron> tetrahedra = new List<Tetrahedron>();
        tetrahedra.Add(new Tetrahedron(points[0], points[1], points[2], points[3]));

        // 2. Insert remaining points
        for (int i = 4; i < points.Length; i++)
        {
            Vector3 pointToInsert = points[i];

            // Find nearby points using KDTree (Important change)
            List<Vector3> nearbyPoints = FindNearbyPoints(kdTree, pointToInsert, points, 3); // Find 3 nearest points


            List<Tetrahedron> possibleTetrahedra = new List<Tetrahedron>();
            foreach (Tetrahedron tetrahedron in tetrahedra)
            {
                foreach(Vector3 nearPoint in nearbyPoints)
                {
                    if (tetrahedron.HasVertex(nearPoint))
                    {
                        possibleTetrahedra.Add(tetrahedron);
                        break;
                    }
                }
            }

            List<Tetrahedron> badTetrahedra = new List<Tetrahedron>();
            List<Triangle> faces = new List<Triangle>();

            foreach (Tetrahedron t in possibleTetrahedra) // Only check nearby tetrahedra
            {
                if (t.IsPointInsideCircumsphere(pointToInsert))
                {
                    badTetrahedra.Add(t);
                    faces.Add(new Triangle(t.a, t.b, t.c));
                    faces.Add(new Triangle(t.a, t.b, t.d));
                    faces.Add(new Triangle(t.a, t.c, t.d));
                    faces.Add(new Triangle(t.b, t.c, t.d));
                }
            }

            tetrahedra.RemoveAll(t => badTetrahedra.Contains(t));

            for (int j = 0; j < faces.Count; j++)
            {
                for (int k = j + 1; k < faces.Count; k++)
                {
                    if (faces[j].Equals(faces[k]))
                    {
                        faces.RemoveAt(k);
                        faces.RemoveAt(j);
                        j--;
                        break;
                    }
                }
            }

            foreach (Triangle f in faces)
            {
                tetrahedra.Add(new Tetrahedron(f.a, f.b, f.c, pointToInsert));
            }
        }

        // 4. Create Mesh (extract boundary triangles)
        // Mesh Creation (Corrected and improved)
        Mesh mesh = new Mesh();
        HashSet<Triangle> boundaryTriangles = new HashSet<Triangle>();

        foreach (Tetrahedron tetrahedron in tetrahedra)
        {
            boundaryTriangles.Add(new Triangle(tetrahedron.a, tetrahedron.b, tetrahedron.c));
            boundaryTriangles.Add(new Triangle(tetrahedron.a, tetrahedron.b, tetrahedron.d));
            boundaryTriangles.Add(new Triangle(tetrahedron.a, tetrahedron.c, tetrahedron.d));
            boundaryTriangles.Add(new Triangle(tetrahedron.b, tetrahedron.c, tetrahedron.d));
        }

        List<Vector3> vertices = new List<Vector3>();
        List<int> triangles = new List<int>();

        foreach (Triangle triangle in boundaryTriangles)
        {
            int v1 = AddVertex(vertices, triangle.a);
            int v2 = AddVertex(vertices, triangle.b);
            int v3 = AddVertex(vertices, triangle.c);

            triangles.Add(v1);
            triangles.Add(v2);
            triangles.Add(v3);
        }

        mesh.vertices = vertices.ToArray();
        mesh.triangles = triangles.ToArray();
        mesh.RecalculateNormals();

        return mesh;
    }

    private int AddVertex(List<Vector3> vertices, Vector3 vertex)
    {
        int index = vertices.FindIndex(v => v.Equals(vertex));
        if (index == -1)
        {
            vertices.Add(vertex);
            return vertices.Count - 1;
        }
        return index;
    }

    private List<Vector3> FindNearbyPoints(DataStructures.ViliWonka.KDTree.KDTree tree, Vector3 position, Vector3[] originalPoints, int numNearest)
    {
        DataStructures.ViliWonka.KDTree.KDQuery query = new DataStructures.ViliWonka.KDTree.KDQuery();
        List<int> results = new List<int>();
        query.KNearest(tree, position, numNearest, results);
        List<Vector3> nearestPoints = new List<Vector3>();
        foreach(int index in results)
        {
            nearestPoints.Add(originalPoints[index]);
        }
        return nearestPoints;
    }

    private Bounds CalculateBounds(Vector3[] points)
    {
        if (points == null || points.Length == 0)
            return new Bounds(Vector3.zero, Vector3.zero);

        Bounds bounds = new Bounds(points[0], Vector3.zero);
        foreach (Vector3 point in points)
        {
            bounds.Encapsulate(point);
        }
        return bounds;
    }

    private class Triangle
    {
        public Vector3 a, b, c;

        public Triangle(Vector3 a, Vector3 b, Vector3 c)
        {
            this.a = a;
            this.b = b;
            this.c = c;
        }

        public bool IsPointInsideCircumcircle(Vector3 p)
        {
            float ax = a.x - p.x;
            float ay = a.z - p.z;
            float bx = b.x - p.x;
            float by = b.z - p.z;
            float cx = c.x - p.x;
            float cy = c.z - p.z;

            float det = (ax * (by * (cx * cx + cy * cy) - cy * (bx * bx + by * by)))
                      - (ay * (bx * (cx * cx + cy * cy) - cx * (bx * bx + by * by)))
                      + ((ax * by - ay * bx) * (bx * cy - by * cx));

            return det < 0;
        }

        public bool SharesVertex(Vector3 v)
        {
            return a == v || b == v || c == v;
        }
    }

    private class Edge
    {
        public Vector3 p1, p2;

        public Edge(Vector3 p1, Vector3 p2)
        {
            this.p1 = p1;
            this.p2 = p2;
        }

        public override bool Equals(object obj)
        {
            if (obj == null || GetType() != obj.GetType())
            {
                return false;
            }

            Edge other = (Edge)obj;
            return (p1 == other.p1 && p2 == other.p2) || (p1 == other.p2 && p2 == other.p1);
        }

        public override int GetHashCode()
        {
            return p1.GetHashCode() ^ p2.GetHashCode();
        }
    }

    private class Tetrahedron
    {
        public Vector3 a, b, c, d;

        public Tetrahedron(Vector3 a, Vector3 b, Vector3 c, Vector3 d)
        {
            this.a = a;
            this.b = b;
            this.c = c;
            this.d = d;
        }

        public bool IsPointInsideCircumsphere(Vector3 p)
        {
            Matrix4x4 m = new Matrix4x4();
            m.SetColumn(0, new Vector4(a.x - p.x, b.x - p.x, c.x - p.x, d.x - p.x));
            m.SetColumn(1, new Vector4(a.y - p.y, b.y - p.y, c.y - p.y, d.y - p.y));
            m.SetColumn(2, new Vector4(a.z - p.z, b.z - p.z, c.z - p.z, d.z - p.z));
            m.SetColumn(3, new Vector4((a - p).sqrMagnitude, (b - p).sqrMagnitude, (c - p).sqrMagnitude, (d - p).sqrMagnitude));

            return m.determinant > Mathf.Epsilon;
        }
        public bool SharesVertex(Vector3 v)
        {
            return a == v || b == v || c == v || d == v;
        }

        public bool HasVertex(Vector3 vertex)
        {
            return this.a.Equals(vertex) || this.b.Equals(vertex) || this.c.Equals(vertex) || this.d.Equals(vertex);
        }
    }
}